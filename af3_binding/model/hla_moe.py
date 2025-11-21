import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve,average_precision_score
from argparse import ArgumentParser,Namespace
import matplotlib.pyplot as plt
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,FixedAbsolutePositionEmbedding
from mixture_of_experts import MoE
import seaborn as sns
import math
import random
import itertools
from visualizer import get_local
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MOE_MHC(nn.Module):

    def __init__(self, hparams):
        super(MOE_MHC, self).__init__()
        hparams = Namespace(**hparams)
        configs = hparams
        self.configs = configs
        # Model Type
        self.tcr_encoding_model = hparams.tcr_encoding_model
        self.use_alpha = hparams.use_alpha
        self.use_vj = hparams.use_vj
        self.use_mhc = hparams.use_mhc
        self.use_t_type = hparams.use_t_type
        self.cat_encoding = hparams.cat_encoding
        # Dimensions
        self.aa_embedding_dim = hparams.aa_embedding_dim
        self.cat_embedding_dim = hparams.cat_embedding_dim
        self.lstm_dim = hparams.lstm_dim
        self.encoding_dim = hparams.encoding_dim
        self.dropout_rate = hparams.dropout
        self.lr = hparams.lr
        self.wd = hparams.wd

        self.all_encoder = DataEmbedding(hparams.d_model, hparams.d_model, hparams.embed, hparams.freq, hparams.dropout)
   
        self.tcr_emb = FixedAbsolutePositionEmbedding(22, hparams.d_model, hparams.embed, hparams.freq, hparams.dropout)
        self.pep_emb = FixedAbsolutePositionEmbedding(22, hparams.d_model, hparams.embed, hparams.freq, hparams.dropout)
        self.tcra_emb = FixedAbsolutePositionEmbedding(22, hparams.d_model, hparams.embed, hparams.freq, hparams.dropout)
        self.tcrb_emb = FixedAbsolutePositionEmbedding(22, hparams.d_model, hparams.embed, hparams.freq, hparams.dropout)
        self.cat_emb = DataEmbedding(22, hparams.d_model, hparams.embed, hparams.freq, hparams.dropout)
        self.add_pos = FixedAbsolutePositionEmbedding(22, hparams.d_model, hparams.embed, hparams.freq, hparams.dropout,emb_x=False)


        self.cls_fianl = nn.Sequential(
            nn.Linear(hparams.d_model*76,1000),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(1000),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(100),
            nn.Linear(100, 1)
        )
            
        self.cls_ap = nn.Sequential(
            nn.Linear(hparams.d_model*34,1000),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(1000),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(100),
            nn.Linear(100, 1)
        )

        self.cls_bp = nn.Sequential(
            nn.Linear(hparams.d_model*28,1000),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(1000),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(100),
            nn.Linear(100, 1)
        )

        self.layer_norm = RMSNorm(hparams.d_model)
        self.norm_hidden = RMSNorm(1000)
        self.norm_output = RMSNorm(100)
        self.hidden_layer1 = nn.Linear(hparams.d_model*76,1000 )
        self.relu = torch.nn.PReLU()
        self.hidden_output = nn.Linear(1000, 100)
        self.output_layer1 = nn.Linear(100, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder_ap = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads)
        
        self.encoder_bp = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads)
        
        self.encoder_all = AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=True),
                        configs.d_model, configs.n_heads)
        
        self.encoder_a = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=True),
                        configs.d_model, configs.n_heads)
        
        self.encoder_b = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=True),
                        configs.d_model, configs.n_heads)
        
        self.encoder_p = AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=True),
                        configs.d_model, configs.n_heads)
        
        self.moe = MoE(
                dim = hparams.d_model,
                num_experts = 16,               # increase the experts (# parameters) of your model without increasing computation
                hidden_dim = hparams.d_model,           # size of hidden dimension in each expert, defaults to 4 * dimension
                activation = nn.GELU,      # use your preferred activation, will default to GELU
                second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
                second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
                second_threshold_train = 0.2,
                second_threshold_eval = 0.2,
                capacity_factor_train = 2,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
                )

        self.moe_ablation =  MoE(
                dim = hparams.d_model,
                num_experts = 1,               # increase the experts (# parameters) of your model without increasing computation
                hidden_dim = hparams.d_model ,           # size of hidden dimension in each expert, defaults to 4 * dimension
                activation = nn.GELU,      # use your preferred activation, will default to GELU
                second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
                second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
                second_threshold_train = 0.2,
                second_threshold_eval = 0.2,
                capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
                )
        
        self.moe_hook = None
        self.cls_hook = None
        self.all_hook = None
        self.bcls_hook = None

        self.hla_type = 54+1
        self.hla_net = nn.Sequential(
            nn.Linear(hparams.d_model*76 , 1000),
            nn.ReLU(),
            nn.Linear(1000,  self.hla_type)  # 输出层节点数 = 类别数
        )
        self.hla_loss =nn.CrossEntropyLoss()
        self.hla_enc = nn.Linear(176,10)
        self.hla_linears = nn.ModuleList([nn.Linear(100+5,1) for i in range( self.hla_type)])
        self.output_layer2 = nn.Linear(self.hla_type, 1)

        self.hla_cls = nn.ModuleList( nn.Sequential(
            nn.Linear(hparams.d_model*76,1000),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(1000),
            nn.Linear(1000,100),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            RMSNorm(100),
            nn.Linear(100, 1)
        )  for i in range( self.hla_type))

        self.a_projection = nn.Linear(hparams.d_model*34, hparams.d_model, bias=False)
        self.b_projection = nn.Linear(hparams.d_model*28, hparams.d_model, bias=False)
        self.p_projection = nn.Linear(hparams.d_model*14, hparams.d_model, bias=False)

        #for 9.10 ver
        self.co_linear = nn.Linear(128, 64, bias=False)
        # for 6.23 ver 
        # self.co_linear = nn.Linear(76, 40, bias=False)

        # for 6.27 ver
        # self.co_linear = nn.Linear(128, 64, bias=False)

        # for 9.22 ver
        # self.align = nn.Linear(128, 128, bias=False)

    
    @get_local('seq_attn')   
    def forward(self, 
                seq_tokens:torch.Tensor,
                embedding_single:torch.Tensor,
                embedding_pair:torch.Tensor,
                chain_encoding:torch.Tensor,
                distance_embedding:torch.Tensor,
                pae_embedding:torch.Tensor,
                plddts:torch.Tensor,
                va_token:torch.Tensor,
                ja_token:torch.Tensor,
                vb_token:torch.Tensor,
                jb_token:torch.Tensor,
                hla_token:torch.Tensor,
                cdr3a_tokens:torch.Tensor,
                cdr3b_tokens:torch.Tensor,
                peptide_tokens:torch.Tensor,
                **kwargs):
        
        tcra = F.one_hot(cdr3a_tokens,num_classes=22).float()
        tcra = F.pad(tcra, (0, 0, 0, 34 - tcra.shape[1]), value=0.0)  # pad to 34
        tcrb = F.one_hot(cdr3b_tokens,num_classes=22).float() 
        tcrb = F.pad(tcrb, (0, 0, 0, 28 - tcrb.shape[1]), value=0.0)  # pad to 28
        pep = F.one_hot(peptide_tokens,num_classes=22).float() 
        pep = F.pad(pep, (0, 0, 0, 14 - pep.shape[1]), value=0.0)  # pad to 14
        va = va_token
        vb = vb_token
        ja = ja_token
        jb = jb_token
        mhc = hla_token


        #record valid pos
        tcra_pos = self.pair_notnone(cdr3a_tokens)
        tcrb_pos = self.pair_notnone(cdr3b_tokens)
        pep_pos = self.pair_notnone(peptide_tokens)



        #build pad_mark 
        tcra_mark = (tcra.sum(dim=-1) != 0).float()
        tcrb_mark = (tcrb.sum(dim=-1) != 0).float()
        pep_mark = (pep.sum(dim=-1) != 0).float()

        #mask
        if self.training:
            tcra_mark = self.masked_replace(tcra_mark)
            tcrb_mark = self.masked_replace(tcrb_mark)
            pep_mark = self.masked_replace(pep_mark)

        #emb
        tcr_batch = self.tcr_emb(torch.cat([tcra,tcrb],1), None)
        pep_batch = self.pep_emb(pep, None)
        tcra_batch = self.tcra_emb(tcra, None)
        tcrb_batch = self.tcrb_emb(tcrb, None)
        # va, vb, ja, jb, mhc = cat_batch
        va = va.unsqueeze(1)
        vb = vb.unsqueeze(1)
        ja = ja.unsqueeze(1)
        jb = jb.unsqueeze(1)
        mhc = mhc.unsqueeze(1)
        cat_batch = torch.cat([va,vb,ja,jb,mhc],1)

        #add pe
        d_model = tcr_batch.shape[2]
        pe = torch.zeros(100, d_model).float()
        pe.require_grad = False
        position = torch.tensor([0]*34+[1]*28+[2]*14+[3]*24).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()* -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe[:, :76]
        # tcra_batch = tcra_batch + pe[:,:34]
        # tcrb_batch = tcrb_batch[:,34:34+28] + pe[:,34:34+28]
        # pep_batch = pep_batch + pe[:,34+28:34+28+14]        

        #
        tcra_batch = self.layer_norm(self.encoder_a(tcra_batch,tcra_batch,tcra_batch, attn_mask=None)[0] + tcra_batch)
        tcrb_batch = self.layer_norm(self.encoder_b(tcrb_batch,tcrb_batch,tcrb_batch, attn_mask=None)[0] + tcrb_batch)
        pep_batch = self.layer_norm(self.encoder_p(pep_batch,pep_batch,pep_batch, attn_mask=None)[0] + pep_batch)

    
        # print(tcra_batch.shape,tcra_mark.shape)
        # print(tcrb_batch[0],tcra_mark[0])
        #mask
        tcra_batch = tcra_batch * tcra_mark.unsqueeze(2)
        tcrb_batch = tcrb_batch * tcrb_mark.unsqueeze(2)
        pep_batch = pep_batch * pep_mark.unsqueeze(2)
        #cat
        all = torch.cat([tcra_batch,tcrb_batch,pep_batch],1)


        seq_attn = self.encoder_all(all,all,all, attn_mask=None)[1]
        #10.27 ver
        # all = self.encoder_all(all,all,all,attn_mask=None)[0]
        #for multi-model
        
        # for 9.23 ver
        single_embedding = all
        # print(single_embedding.mean(-1))
        try:
            single_embedding = self.valid_detach(single_embedding,tcra_pos,tcrb_pos,pep_pos) # [batchsize,max_pos,56]
        except:
            print(cdr3a_tokens[0],cdr3b_tokens[0],peptide_tokens[0],pep_pos)
        # print(single_embedding.mean(-1))
        single_embedding = self.co_linear(single_embedding)  # [batchsize,76,56]        
        
        # 6.23 ver
        # single_embedding = self.valid_detach(single_embedding,tcra_pos,tcrb_pos,pep_pos) # [batchsize,max_pos,56]
        # 9.22 ver
        # single_embedding =  self.align(single_embedding)

        #6.27 ver
        # single_embedding = self.co_linear(single_embedding)

        # single_embedding = self.co_linear(all.permute(0,2,1)).permute(0,2,1)  # [batchsize,76,40]

        # print(single_embedding.shape)
        self.all_hook = all
        # print(tcra_batch.shape,pep_batch.shape)
        #
        #task 2: a and b 
        a_encoder = self.a_projection(tcra_batch.reshape(-1,tcra_batch.shape[1]*tcra_batch.shape[2]))
        b_encoder = self.b_projection(tcrb_batch.reshape(-1,tcrb_batch.shape[1]*tcrb_batch.shape[2]))
        p_encoder = self.p_projection(pep_batch.reshape(-1,pep_batch.shape[1]*pep_batch.shape[2]))
        # print(b_encoder.size()) # [batchsize,200]

        # Calculating the Loss
        a_encoder = a_encoder / a_encoder.norm(p=2, dim=-1, keepdim=True)
        b_encoder = b_encoder / b_encoder.norm(p=2, dim=-1, keepdim=True)
        p_encoder = p_encoder / p_encoder.norm(p=2, dim=-1, keepdim=True)
        logits = (a_encoder @ b_encoder.T) /0.5
        clip_loss = 0.3*self.clip_loss(logits)
        logits = ((a_encoder+b_encoder) @ p_encoder.T) /0.5
        clip_loss += 0.3*self.clip_loss(logits)
        # logits = (b_encoder @ p_encoder.T) /0.2
        # clip_loss += 0.2*self.clip_loss(logits)

        aux_loss = 0
        #MOE layer
        moe_out, aux_loss , combine_tensor= self.moe(all)

        all = self.layer_norm(moe_out + all)


        #10.27 ver
        # single_embedding = all
        # single_embedding = self.valid_detach(single_embedding,tcra_pos,tcrb_pos,pep_pos) # [batchsize,max_pos,56]
        # single_embedding = self.co_linear(single_embedding)  # [batchsize,76,56]        

        if not self.training:
            if self.moe_hook==None:
                # print('record')
                self.moe_hook = combine_tensor.mean(0).mean(2)
            else:
                # print('update')
                self.moe_hook += combine_tensor.mean(0).mean(2)


        all = all.reshape(-1,all.shape[1]*all.shape[2])

        #task 4：prediciton binding score
        self.bcls_hook = all

        hidden_output = self.dropout(self.relu(self.hidden_layer1(all)))        
        hidden_output = self.norm_hidden(hidden_output) 
        hidden_output = self.dropout(self.relu(self.hidden_output(hidden_output)))
        hidden_output = self.norm_output(hidden_output)

        outputs = torch.zeros(hidden_output.size(0), 1, device=hidden_output.device)  # 初始化输出容器            
     
        hidden_output = torch.cat([hidden_output,cat_batch],1)

        self.cls_hook = hidden_output

        for head_idx in range(len(self.hla_linears)):
            mask = (mhc.squeeze() == head_idx)
            if mask.any():
                # 将对应数据送入指定头
                selected_features = hidden_output[mask]
                outputs[mask] = self.hla_linears[head_idx](selected_features)

        mlp_output = outputs.squeeze()

        # output = torch.sigmoid(mlp_output.squeeze()) #* torch.sigmoid(abp.squeeze())
        output = mlp_output.squeeze() 
        
        return output,aux_loss + clip_loss,single_embedding
    
    def contrastive_loss(self,logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self,similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    def masked_replace(self,
                       seq: torch.Tensor,
                       mask_token: int = 0,  # 假设原始氨基酸索引为0-20，21是掩码符号
                    mask_prob: float = 0.15,
                    pad_token: int = 0) -> torch.Tensor:
        """
        对输入序列进行随机掩码替换（排除填充位置）
        
        Args:
            seq (torch.Tensor): 输入序列，形状可为 (B, L) 或 (L,)
            mask_token (int): 掩码符号的索引（需确保不与原始数据冲突）
            mask_prob (float): 每个有效位置被掩码的概率
            pad_token (int): 填充符号的索引
        
        Returns:
            torch.Tensor: 掩码后的序列
        """
        
        if mask_prob <= 0:
            return seq
        
        # 生成随机掩码矩阵（与输入同形状）
        rand_mask = torch.rand_like(seq, dtype=torch.float)
        
        # 生成有效位置掩码（排除填充）
        valid_positions = (seq != pad_token)
        
        # 合并掩码条件：随机概率达标且是有效位置
        final_mask = (rand_mask < mask_prob) & valid_positions
        
        # 应用掩码
        masked_seq = seq.clone()
        masked_seq[final_mask] = mask_token
        
        return masked_seq
    def pair_notnone(self, seq: torch.Tensor,) -> torch.Tensor:
            # Step 1: 创建非零掩码 (形状: D0 x D1 x D2)
        mask = (seq != 0)
        # Step 2: 反转第二维
        mask_flipped = torch.flip(mask, dims=[1])
        
        # Step 3: 查找反转后第一个非零位置 (形状: D0 x D2)
        rev_idx = mask_flipped.int().argmax(dim=1)
        
        # Step 4: 计算原始索引
        d1 = seq.size(1)
        orig_idx = d1 - rev_idx

        # Step 5: 处理全零切片
        any_nonzero = mask.any(dim=1)  # 检查是否存在非零元素
        result = torch.where(any_nonzero, orig_idx, torch.tensor(-1, device=seq.device))        
        # print(seq,result)
        return result
    def valid_detach(self,seq,pos1,pos2,pos3):
        max_pos = (pos1+pos2+pos3).max()
        copy_seq = torch.zeros(seq.shape[0], max_pos, seq.shape[2], device=seq.device)
        for i in range(seq.shape[0]):
            if pos3[i] > 14:
                pos3[i] = 14
            copy_seq[i,:pos1[i],:] = seq[i,:pos1[i],:]
            copy_seq[i,pos1[i]:pos1[i]+pos2[i],:] = seq[i,34:34+pos2[i],:]
            # try:
            copy_seq[i,pos1[i]+pos2[i]:pos1[i]+pos2[i]+pos3[i],:] = seq[i,62:62+pos3[i],:]
            # except:
                # print(i,pos1[i],pos2[i],pos3[i],copy_seq,seq)
        return copy_seq

    
    def plot_moe(self,save_path=None,seperate = True):
        # print(self.moe_hook)
        # 创建画布

        
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'

        print(self.moe_hook.shape)
        plt.clf()
        if seperate:
            fig, ax = plt.subplots(figsize=(9, 6))
            # 绘制热图
            sns.heatmap(
                self.moe_hook[:34].cpu().T,
                annot=None,          # 显示数值
                fmt=".2f",           # 数值格式
                cmap="RdPu",     # 颜色映射
                linewidths=0.5,      # 单元格边框线宽
                linecolor="black",   # 边框颜色
                ax=ax,
                cbar_kws={"shrink": 0.8}  # 颜色条设置
            )

            # 添加标题和坐标轴标签
            ax.set_title("The use of experts each TCRα Amino Acid")
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Experts")

            # 调整布局
            plt.tight_layout()
            if save_path is not None:
                print('save_plot moe')
                plt.savefig("moe_fig/" + save_path + "tcra.svg",format='svg', dpi=300, bbox_inches='tight')
            else:
                plt.savefig("moe_fig/tcra.png", dpi=300, bbox_inches='tight')
            fig.clf()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                self.moe_hook[34:34+28].cpu().T,
                annot=None,          # 显示数值
                fmt=".2f",           # 数值格式
                cmap="RdPu",     # 颜色映射
                linewidths=0.5,      # 单元格边框线宽
                linecolor="black",   # 边框颜色
                ax=ax,
                cbar_kws={"shrink": 0.8}  # 颜色条设置
            )

            # 添加标题和坐标轴标签
            ax.set_title("The use of experts TCRβ Amino Acid")
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Experts")

            # 调整布局
            plt.tight_layout()
            if save_path is not None:
                print('save_plot moe')
                plt.savefig("moe_fig/" + save_path + "tcrb.svg",format='svg', dpi=300, bbox_inches='tight')
            else:
                plt.savefig("moe_fig/tcrb.png", dpi=300, bbox_inches='tight')
            fig.clf()
            fig, ax = plt.subplots(figsize=(4, 6))
            sns.heatmap(
                self.moe_hook[62:].cpu().T,
                annot=None,          # 显示数值
                fmt=".2f",           # 数值格式
                cmap="RdPu",     # 颜色映射
                linewidths=0.5,      # 单元格边框线宽
                linecolor="black",   # 边框颜色
                ax=ax,
                cbar_kws={"shrink": 0.8}  # 颜色条设置
            )

            # 添加标题和坐标轴标签
            ax.set_title("The use of experts each peptide Amino Acid")
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Experts")

            # 调整布局
            plt.tight_layout()
            if save_path is not None:
                print('save_plot moe')
                plt.savefig("moe_fig/" + save_path + "pep.svg",format='svg', dpi=300, bbox_inches='tight')
            else:
                plt.savefig("moe_fig/pep.png", dpi=300, bbox_inches='tight')
        else:
            fig, ax = plt.subplots(figsize=(21, 6))
            # 绘制热图
            sns.heatmap(
                self.moe_hook.cpu().T,
                annot=None,          # 显示数值
                fmt=".2f",           # 数值格式
                cmap="coolwarm",     # 颜色映射
                linewidths=0.5,      # 单元格边框线宽
                linecolor="black",   # 边框颜色
                ax=ax,
                cbar_kws={"shrink": 0.8}  # 颜色条设置
            )

            # 添加标题和坐标轴标签
            ax.set_title("The use of experts each Amino Acid")
            ax.set_xlabel("Tokens")
            ax.set_ylabel("Experts")

            # 调整布局
            plt.tight_layout()
            if save_path is not None:
                print('save_plot moe')
                plt.savefig("moe_fig/" + save_path + "heatmap.svg", dpi=300, bbox_inches='tight',format = 'svg')
            else:
                plt.savefig("moe_fig/heatmap.png", dpi=300, bbox_inches='tight')
