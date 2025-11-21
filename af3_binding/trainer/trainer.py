import os
import os.path as osp
import time

import yaml
import random
import numpy as np
import logging
import logging.config

import torch
import torch.nn as nn
from sklearn import metrics

from af3_binding.data import IEDBDataLoader, get_train_and_test_dataset
from af3_binding.model import Model

class BDtrainer:
    def __init__(self, config):
        self.config = config
        self._prepare()
        self.device = torch.device(config['device'])
        self.num_epochs = config['num_epochs']
        self.seed = config['seed']
        self._set_seed()

        self.train_set, self.test_set = get_train_and_test_dataset(**config['dataset'])
        self.train_loader = IEDBDataLoader(self.train_set, **config['dataloader'])
        self.test_loader = IEDBDataLoader(self.test_set, **config['dataloader'])

        # 添加MLM相关配置
        mlm_config = config.get('mlm_config', None)
        self.model = Model(
            feature_config=config['model']['feature_config'],
            model_config=config['model']['model_config'],
            hla_config=config['model']['hla_config'],
            mlm_config=config['model']['mlm_config'],
        ).to(self.device)
        
        self.best_auc = 0.0  # 添加变量跟踪最佳AUC
        self.best_epoch = -1  # 记录最佳AUC对应的epoch
        self.best_aupr = 0.0
        self.best_aupr_epoch = -1
        # 模型创建后
        first_block = self.model.pairformer.blocks[0]
        num_blocks = len(self.model.pairformer.blocks)
        print("\n==== 配置参数验证 ====")
        print(f"配置中use_single_embed: {config['model']['feature_config']['use_single_embed']} | 实际use_single_embed: {self.model.features.use_single_embed}")
        print(f"配置中use_pair_embed: {config['model']['feature_config']['use_pair_embed']} | 实际use_pair_embed: {self.model.features.use_pair_embed}")
        print(f"配置中num_blocks: {config['model']['model_config']['num_blocks']} | 实际num_blocks: {num_blocks}")
        # print(f"配置中dim: {config['model']['model_config']['dim']} | 实际dim: {first_block.pair_update.tri_mul_out.dim}")
        print(f"配置中num_heads: {config['model']['model_config']['num_heads']} | 实际num_heads: {first_block.single_update.attn.num_heads}")
        print(f"配置中dropout: {config['model']['model_config']['dropout']} | 实际dropout率: {first_block.pair_update.dropout_row.dropout.p}")
        # 添加MLM配置的简洁验证（与其他参数验证格式一致）
        print(f"是否MLM: {self.model.do_mlm}")
        if self.model.do_mlm:
            print(f"配置中MLM权重: {config['model']['mlm_config']['weight']} | 实际MLM权重: {self.model.mlm_weight}")
            print(f"配置中掩码概率: {config['model']['feature_config']['mask_prob']} | 实际掩码概率: {self.model.features.mask_prob}")


        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f'total trainable params: {total_params}')
        logging.info(f'model: {self.model}')

        # 损失函数
        # self.binding_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(self.device))  # 训练集负:正 = 3:1
        self.binding_criterion = nn.BCEWithLogitsLoss()
        if self.model.do_mlm:
            self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略padding
        self.optimizer = torch.optim.Adam(self.model.parameters(), **config['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **config['scheduler'])


    def _prepare(self):
        # prepare out folders
        self.out_dir = osp.join(self.config['output_dir'], self.config['name'])
        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if not osp.exists(osp.join(self.out_dir, 'model_weights')):
            os.makedirs(osp.join(self.out_dir, 'model_weights'))
        with open(osp.join(self.out_dir, 'config.yml'), 'w') as f:
            yaml.safe_dump(self.config, f)
        
        # update logging
        loggingConfigDict = self.config['logger']
        for _, handler in loggingConfigDict['handlers'].items():
            if 'filename' in handler.keys():
                handler['filename'] = osp.join(self.out_dir, handler['filename'])
        logging.config.dictConfig(loggingConfigDict)

    def _set_seed(self):
        if self.seed:
            self.seed = int(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def train(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            epoch_binding_loss = 0
            epoch_mlm_loss = 0
            epoch_batch = 0
            self.model.train()
            t1 = time.time()
            for i, batch in enumerate(self.train_loader):
                t0 = time.time()
                inputs = batch
                # print(batch['seq_tokens'][0].shape,batch['cdr3b_tokens'][0].shape,batch['cdr3a_tokens'][0].shape)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = batch['label'].float()
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                bind_loss = self.binding_criterion(outputs['binding_pred'], labels)
                loss= bind_loss + outputs['sup_loss']
                mlm_loss = 0
                if self.model.do_mlm:
                    mlm_logits = outputs['mlm_logits']  # [B, L, aa_size]
                    mlm_labels = outputs['mlm_labels']  # [B, L]
                    mask_indices = outputs['mask_indices']  # [B, L]
                # 仅对掩码位置计算MLM损失
                    # B, L, C = mlm_logits.shape
                    # mlm_logits_flat = mlm_logits.reshape(-1, C)  # [B*L, aa_size]
                    # mlm_labels_flat = mlm_lables.reshape(-1)   # [B*L]
                    mask_indices_flat = mask_indices.reshape(-1)  # [B*L]
                    
                    if mask_indices_flat.sum() > 0:  # 确保有掩码位置
                    #     mlm_loss = self.mlm_criterion(
                    #         mlm_logits_flat[mask_indices_flat], 
                    #         mlm_labels_flat[mask_indices_flat],
                    #     )
                        mlm_logits_flat = mlm_logits.view(-1, mlm_logits.size(-1))
                        mlm_labels_flat = mlm_labels.view(-1) # [B*L]
                        mlm_loss = self.mlm_criterion(mlm_logits_flat, mlm_labels_flat)
                        loss = bind_loss + self.model.mlm_weight * mlm_loss  
                

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_binding_loss += bind_loss.item()
                if self.model.do_mlm:
                    epoch_mlm_loss += mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else 0
                epoch_batch += 1
                

                if i % 10 == 0:
                    logging.info(f'Epoch {epoch:>4}, Batch {i:>4}, Loss: {loss.item():>2.4f},Bind Loss: {bind_loss.item()},MLM Loss: {mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else 0},Collect data time: {t0-t1:>3.1f}s, Training time: {time.time()-t0:>3.1f}s')
                t1 = time.time()
            
            avg_loss = epoch_loss / epoch_batch
            average_binding_loss = epoch_binding_loss / epoch_batch
            average_mlm_loss = epoch_mlm_loss / epoch_batch if self.model.do_mlm else 0
            logging.info(f'Epoch {epoch}, Average Loss: {avg_loss}, '
                        f'Average Binding Loss: {average_binding_loss}, '
                        f'Average MLM Loss: {average_mlm_loss if self.model.do_mlm else 0}')
            self.scheduler.step()
            auc, aupr = self.test()
            if auc > self.best_auc:
                self.best_auc = auc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), osp.join(self.out_dir, 'model_weights', 'best_auc_model.pt'))
                logging.info(f'Saved best AUC model at epoch {epoch} with AUC: {auc:.4f}')
            
            # 保存最佳 AUPR 模型
            if aupr > self.best_aupr:
                self.best_aupr = aupr
                self.best_aupr_epoch = epoch
                torch.save(self.model.state_dict(), osp.join(self.out_dir, 'model_weights', 'best_aupr_model.pt'))
                logging.info(f'Saved best AUPR model at epoch {epoch} with AUPR: {aupr:.4f}')
            if (epoch+1) % 1 == 0:
                torch.save(self.model.state_dict(), osp.join(self.out_dir, 'model_weights', f'epoch_{epoch+1}.pt'))
            #save the best auc model
            
                

    def test(self):
        self.model.eval()
        all_pred = []
        all_label = []

        if self.model.do_mlm:
            all_mlm_preds = []
            all_mlm_targets = []
        with torch.no_grad():
            total_loss = 0
            total_binding_loss = 0
            total_mlm_loss = 0
            total_num = 0
            for i, batch in enumerate(self.test_loader):
                inputs = batch
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = batch['label'].float()
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                
                # 主任务评估
                binding_loss = self.binding_criterion(outputs['binding_pred'], labels)
                loss = binding_loss + outputs['sup_loss']
                
                # MLM任务评估
                mlm_loss = 0
                if self.model.do_mlm and 'mlm_logits' in outputs:
                    mlm_logits = outputs['mlm_logits']
                    orig_tokens = outputs['mlm_labels'] # 使用 mlm_labels 而非 orig_tokens
                    mask_indices = outputs['mask_indices']
                    
                    B, L, C = mlm_logits.shape
                    mlm_logits_flat = mlm_logits.reshape(-1, C)
                    orig_tokens_flat = orig_tokens.reshape(-1)
                    mask_indices_flat = mask_indices.reshape(-1)
                    
                    if mask_indices_flat.sum() > 0:
                        mlm_loss = self.mlm_criterion(
                            mlm_logits_flat[mask_indices_flat], 
                            orig_tokens_flat[mask_indices_flat]
                        )
                        loss = binding_loss + self.model.mlm_weight * mlm_loss

                    # 计算MLM准确率
                    mlm_preds = mlm_logits.argmax(dim=-1)  # [B, L]
                    if mask_indices.sum() > 0:
                        all_mlm_preds.append(mlm_preds[mask_indices].cpu().numpy())
                        all_mlm_targets.append(orig_tokens[mask_indices].cpu().numpy())

                total_loss += loss.item() * labels.size(0)
                total_binding_loss += binding_loss.item() * labels.size(0)
                if isinstance(mlm_loss, torch.Tensor):
                    total_mlm_loss += mlm_loss.item() * labels.size(0)
                total_num += labels.size(0)

                # 收集主任务预测结果
                all_label.append(labels.cpu().numpy())
                all_pred.append(torch.sigmoid(outputs['binding_pred']).cpu().numpy())
            
            # 主任务评估
            all_label = np.concatenate(all_label)
            all_pred = np.concatenate(all_pred)
            fpr, tpr, _ = metrics.roc_curve(all_label, all_pred)
            auc = metrics.auc(fpr, tpr)
            # 计算 AUPR
            precision, recall, _ = metrics.precision_recall_curve(all_label, all_pred)
            aupr = metrics.auc(recall, precision)
            
            result_str = f'Test Loss: {total_loss / total_num:.4f}, ' \
                         f'Test Binding Loss: {total_binding_loss / total_num:.4f}, ' \
                         f'Test Binding AUC: {auc:.4f}, Test AUPR: {aupr:.4f}'
            
            # MLM任务评估
            if self.model.do_mlm and all_mlm_preds:
                all_mlm_preds = np.concatenate(all_mlm_preds)
                all_mlm_targets = np.concatenate(all_mlm_targets)
                mlm_accuracy = (all_mlm_preds == all_mlm_targets).mean()
                result_str += f', Test MLM Loss: {total_mlm_loss / total_num:.4f}, ' \
                              f'Test MLM Accuracy: {mlm_accuracy:.4f}'
            
            logging.info(result_str)
            return auc,aupr