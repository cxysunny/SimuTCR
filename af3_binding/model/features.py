import torch
import torch.nn as nn
import numpy as np
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,FixedAbsolutePositionEmbedding
from mixture_of_experts import MoE
import itertools
class PositionEncoding(nn.Module):
    def __init__(self, d_model:int=128, max_len:int=2000) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._get_position_encoding()

    def _get_position_encoding(self) -> torch.Tensor:
        r"""Get position encoding matrix

        Returns:
            position encoding matrix, shape should be (max_len, d_model)
        """
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, positions:torch.Tensor) -> torch.Tensor:
        r"""Add position encoding to input

        Args:
            positions: input, shape should be (B, L)
        Returns:
            output, shape should be (B, L, d_model)
        """
        pe = self.pe.to(positions.device) # [L, d]
        pe = pe[None, :, :].repeat(positions.shape[0], 1, 1) # [B, L, d]
        return pe[torch.arange(positions.shape[0])[:, None], positions, :]
        
class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model:int=128, max_len:int=32) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(2*max_len+1, d_model)
    
    def forward(self, offset:torch.Tensor) -> torch.Tensor:
        r"""Add relative position encoding to input

        Args:
            offset: offset of each atom, shape should be (B, L, K)
        Returns:
            output, shape should be (B, L, K, d_model)
        """
        B, L, _ = offset.shape
        offset = torch.clip(offset + self.max_len, 0, 2*self.max_len)
        return self.embedding(offset)

class Features(nn.Module):
    def __init__(self, aa_size:int, out_dim:int,
                 va_size:int, ja_size:int, va_dim:int, ja_dim:int,
                 vb_size:int, jb_size:int, vb_dim:int, jb_dim:int,
                 hla_size:int, hla_dim:int,
                 use_single_embed:bool=True,use_pair_embed:bool=True,
                 mask_prob:float=0.15):
        super(Features, self).__init__()
        self.use_single_embed = use_single_embed
        self.use_pair_embed = use_pair_embed
        self.mask_token_id = 22
        self.mask_prob = mask_prob
        self.aa_embedding = nn.Embedding(aa_size, 56)
        if self.use_single_embed:
            self.single_compress = nn.Sequential(
                nn.Linear(384, 128),
                nn.GELU(),
                nn.Linear(128, 64),
            )
            single_dim = 56+64+8+16  # 144
        else:
            single_dim = 56+8+16     # 80   
        if self.use_pair_embed:
            self.pair_compress = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 64),
            )
            pair_dim = 64+64+16      # 144
        else:
            pair_dim = 64+16         # 80
        
        self.chain_embedding = nn.Embedding(4, 8)
        self.distance_compress = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        self.pae_compress = nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.plddt_embedding = nn.Embedding(20, 16, padding_idx=0)
        self.va_embedding = nn.Embedding(va_size+1, va_dim, padding_idx=va_size)
        self.ja_embedding = nn.Embedding(ja_size+1, ja_dim, padding_idx=ja_size)
        self.vb_embedding = nn.Embedding(vb_size+1, vb_dim, padding_idx=vb_size)
        self.jb_embedding = nn.Embedding(jb_size+1, jb_dim, padding_idx=jb_size)
        self.hla_embedding = nn.Embedding(hla_size+1, hla_dim, padding_idx=hla_size)

        self.single_position_encoding = PositionEncoding(d_model=single_dim, max_len=400) 
        self.single_position_encoding_with_seq = PositionEncoding(d_model=single_dim+64, max_len=400) 
        self.pair_position_encoding = RelativePositionEncoding(d_model=pair_dim, max_len=400)

        self.single_out = nn.Linear(single_dim, out_dim)
        self.single_out_with_seq = nn.Linear(single_dim+64, out_dim) # for seq embedding

        self.pair_out = nn.Linear(pair_dim, out_dim)
        self.point_out = nn.Linear(va_dim+vb_dim+ja_dim+jb_dim+hla_dim, out_dim)
        
        #modified by jcwu
        self.pep_emb = FixedAbsolutePositionEmbedding(56, 128,0,0, 0.2)
        self.tcra_emb = FixedAbsolutePositionEmbedding(56,128,0,0, 0.2)
        self.tcrb_emb = FixedAbsolutePositionEmbedding(56,128,0,0, 0.2)     

        self.seq_emb = FixedAbsolutePositionEmbedding(56,56,0,0, 0.2) 
        self.dropout = nn.Dropout(p=0.2)

        self.encoder_seq = AttentionLayer(
                        FullAttention(False, 1, attention_dropout=0.2,
                                        output_attention=False),
                        56, 8)
     
        self.moe = MoE(
                dim = 56,
                num_experts = 16,               # increase the experts (# parameters) of your model without increasing computation
                hidden_dim = 56,           # size of hidden dimension in each expert, defaults to 4 * dimension
                activation = nn.GELU,      # use your preferred activation, will default to GELU
                second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
                second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
                second_threshold_train = 0.2,
                second_threshold_eval = 0.2,
                capacity_factor_train = 2,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
                )

        self.seq_empower = False

    def apply_mlm_mask(self, seq_tokens):
        """应用MLM掩码到序列tokens
        
        Args:
            seq_tokens: 原始序列tokens [B, L]
            mask_prob: 掩码概率
            training: 是否处于训练模式
            
        Returns:
            masked_tokens: 掩码后的序列 [B, L]
            mask_indices: 掩码位置 [B, L]
        """
        # if not training:
        #     # 如果不是训练模式，不进行掩码
        #     mask_indices = torch.zeros_like(seq_tokens).bool()
        #     return seq_tokens, mask_indices
        
        # 创建副本，不改变原始输入
        masked_tokens = seq_tokens.clone()
        
        # 仅在有效的非填充位置应用掩码
        valid_tokens = (seq_tokens != 0)  # 排除已经是填充的位置
        
        # 生成随机掩码
        # mask_prob_tensor = torch.full(valid_mask.shape, self.mask_prob, 
        #                              device=seq_tokens.device)
        # mask_indices = torch.bernoulli(mask_prob_tensor).bool() & valid_mask
        
        device = seq_tokens.device
        rand = torch.rand(seq_tokens.shape, device=device)
        mask_indices = (rand < self.mask_prob) & valid_tokens
        # 应用掩码
        masked_tokens[mask_indices] = self.mask_token_id
        mlm_labels = seq_tokens.clone()
        # 将掩码位置的标签设置为-100
        mlm_labels[~mask_indices] = -100
        return masked_tokens, mask_indices, mlm_labels

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
                seq_embed:torch.Tensor,
                do_mlm=False,
                training=True,
                **kwargs):
        B, L = seq_tokens.shape

        # print(seq_tokens.shape,cdr3a_tokens.shape, cdr3b_tokens.shape, peptide_tokens.shape)
        # 保存原始序列用于MLM任务
        orig_seq_tokens = seq_tokens.clone()
        # print(seq_tokens,cdr3a_tokens,cdr3b_tokens, peptide_tokens)
        
        # 如果启用MLM，对序列应用掩码
        if do_mlm:
            # if training:
            seq_tokens, mask_indices, mlm_labels = self.apply_mlm_mask(seq_tokens)
            embedding_single = embedding_single*(~mask_indices.unsqueeze(-1)).float()
            pair_mask_indices = mask_indices.unsqueeze(1) | mask_indices.unsqueeze(2) # [B, L, L]
            embedding_pair = embedding_pair*(~pair_mask_indices.unsqueeze(-1)).float() #
            # else:
               # _, mask_indices, mlm_labels = self.apply_mlm_mask(seq_tokens)
               # seq_tokens, mask_indices, mlm_labels = self.apply_mlm_mask(seq_tokens)
               
        else:
            mask_indices = torch.zeros_like(seq_tokens).bool()
            # mlm_labels = None

        seq_emb = self.aa_embedding(seq_tokens) # [B, L, 56]

        #modified by jcwu
        # if self.seq_empower:
        #     seq_emb = self.seq_emb(seq_emb, None)
        #     seq_emb = self.encoder_seq(seq_emb,seq_emb,seq_emb, attn_mask=None)[0] + seq_emb
        #     moe_out, aux_loss , combine_tensor= self.moe(seq_emb)
        #     seq_emb = moe_out
        #modified end 

        single_mask = (seq_tokens != 0).float() # [B, L]
        # single_mask = ((seq_tokens != 0) & (seq_tokens != self.mask_token_id)).float() # [B, L]
        pair_mask = single_mask[:, None, :] * single_mask[:, :, None]
        if self.use_single_embed:
            single_emb = self.single_compress(embedding_single)  # [B, L, 64]

        chain_emb = self.chain_embedding(chain_encoding) # [B, L, 8]

        plddt_boundary = torch.linspace(0., 100., 20).to(plddts.device) # [1, 20]
        plddt_bins = torch.sum(plddts[..., None] > plddt_boundary, dim=-1) # [B, L]
        plddt_emb = self.plddt_embedding(plddt_bins) # [B, L, 16]
 
        if self.use_pair_embed:
            pair_emb = self.pair_compress(embedding_pair) # [B, L, L, 64]

        distance_emb = self.distance_compress(distance_embedding) # [B, L, L, 64]
        pae_emb = self.pae_compress(pae_embedding) # [B, L, L, 16]

        va_emb = self.va_embedding(va_token) # [B, 32]
        ja_emb = self.ja_embedding(ja_token) # [B, d]
        vb_emb = self.vb_embedding(vb_token) # [B, d]
        jb_emb = self.jb_embedding(jb_token)
        hla_emb = self.hla_embedding(hla_token)
        
        # position encoding
        positions = chain_encoding * 100 + torch.arange(L)[None, :].to(chain_encoding.device) # [B, L]
        positions_pair = positions[:, :, None] - positions[:, None, :] # [B, L, L]

        single_pos_emb = self.single_position_encoding(positions) #[B,L]-> [B,L,144/80]
        pair_pos_emb = self.pair_position_encoding(positions_pair)# [B,L,L]-> [B,L,L,144/80]
        
  
        if seq_embed != None:
            single_embedding = torch.cat([seq_emb, single_emb,seq_embed,chain_emb, plddt_emb], dim=-1) # [B, L, 144+64]
            # print(seq_embed,seq_emb,seq_emb.shape)
            # print(torch.cat([seq_emb, single_emb,chain_emb, plddt_emb], dim=-1))
        elif self.use_single_embed:
            single_embedding = torch.cat([seq_emb, single_emb, chain_emb, plddt_emb], dim=-1) # [B, L, 144]
        else:
            single_embedding = torch.cat([seq_emb, chain_emb, plddt_emb], dim=-1) # [B, L, 80]  

        if self.use_pair_embed:
            pair_embedding = torch.cat([pair_emb, distance_emb, pae_emb], dim=-1) # [B, L, L, 144]
        else:
            pair_embedding = torch.cat([distance_emb, pae_emb], dim=-1) # [B, L, L, 80]    
        
        if seq_embed != None:
            single_pos_emb = self.single_position_encoding_with_seq(positions) # [B, L, 144+64]
            single_embedding = self.single_out_with_seq(single_embedding + single_pos_emb)
        else:
            single_embedding = self.single_out(single_embedding + single_pos_emb)
        pair_embedding = self.pair_out(pair_embedding + pair_pos_emb)
        point_embedding = self.point_out(torch.cat([va_emb, ja_emb, vb_emb, jb_emb, hla_emb], dim=-1)) # [B, d+d+d+d+d]
        
        # 返回额外的MLM信息
        if not do_mlm:
            return (single_embedding, pair_embedding, point_embedding, single_mask, pair_mask)
        else:
            return (single_embedding, pair_embedding, point_embedding, single_mask, pair_mask, orig_seq_tokens, mask_indices, mlm_labels)    
    