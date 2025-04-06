import os
import os.path as osp
import yaml
import random
import numpy as np
import logging
import logging.config

import torch
import torch.nn as nn
from sklearn import metrics

from af3_binding.data import IEDBDataLoader, get_train_and_test_dataset
from af3_binding.model import Model,SimpleModel

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

        self.model = Model(**config['model']).to(self.device)
        # self.model = SimpleModel(**config['model']).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f'total trainable params: {total_params}')
        logging.info(f'model: {self.model}')

        self.criterion = nn.BCEWithLogitsLoss()
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
            epoch_batch = 0

            self.model.train()
            for i, batch in enumerate(self.train_loader):
                inputs = batch
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = batch['label'].float()
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batch += 1
                # if i % 10 == 0:
                #     logging.info(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
                logging.info(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
            #
            average_loss = epoch_loss / epoch_batch
            logging.info(f'Epoch {epoch}, Average Loss: {average_loss}')

            # Test the model
            self.scheduler.step()
            self.test()

            # Save model weights
            if (epoch+1) % 5 == 0:
                torch.save(self.model.state_dict(), osp.join(self.out_dir, 'model_weights', f'epoch_{epoch+1}.pt'))
            # Save the final model
            torch.save(self.model.state_dict(), osp.join(self.out_dir, 'model_weights', 'epoch_last.pt'))    
                

    def test(self):
        self.model.eval()
        all_pred = []
        all_label = []
        with torch.no_grad():
            total_loss = 0
            total_num = 0
            for i, batch in enumerate(self.test_loader):
                inputs = batch
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = batch['label'].float()
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_num += labels.size(0)

                all_label.append(labels.detach().cpu().numpy())
                all_pred.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_label = np.concatenate(all_label)
            all_pred = np.concatenate(all_pred)
            fpr, tpr, _ = metrics.roc_curve(all_label, all_pred)
            auc = metrics.auc(fpr, tpr)
            logging.info(f'Test Loss: {total_loss / total_num}, Test Auc: {auc}')