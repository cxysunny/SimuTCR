import torch
import torch.nn as nn
import numpy as np

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
        position = torch.arange(self.max_len).unsqueeze(1) # 生成[0,1,...,max_len-1]，shape[max_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)) #[d_model/2]
        pe = torch.zeros(self.max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe # [max_len, d_model]

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
        offset = torch.clip(offset + self.max_len, 0, 2*self.max_len) # [B, L, K]
        return self.embedding(offset) # [B,L,K] -> [B,L,K,d_model]

class Features(nn.Module):
    def __init__(self, aa_size:int, out_dim:int,
                 va_size:int, ja_size:int, va_dim:int, ja_dim:int,
                 vb_size:int, jb_size:int, vb_dim:int, jb_dim:int,
                 hla_size:int, hla_dim:int):
        super(Features, self).__init__()
        self.aa_embedding = nn.Embedding(aa_size, 56)
        self.signle_compress = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        self.pair_compress = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
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
        self.plddt_embedding = nn.Embedding(20, 16)
        self.va_embedding = nn.Embedding(va_size, va_dim)
        self.ja_embedding = nn.Embedding(ja_size, ja_dim)
        self.vb_embedding = nn.Embedding(vb_size, vb_dim)
        self.jb_embedding = nn.Embedding(jb_size, jb_dim)
        self.hla_embedding = nn.Embedding(hla_size, hla_dim)
        
        #对单序列特征：d_model = 56（aa_embedding）+64（单链压缩）+8（链嵌入）+16（pLDDT嵌入）=144
        self.single_position_encoding = PositionEncoding(d_model=56+64+8+16, max_len=400) # 144
        #对成对特征：d_model = 64（pair压缩）+64（距离压缩）+16（PAE压缩）=144 
        self.pair_position_encoding = RelativePositionEncoding(d_model=64+64+16, max_len=400)

        self.single_out = nn.Linear(144, out_dim)
        self.pair_out = nn.Linear(144, out_dim)
        self.point_out = nn.Linear(va_dim+ja_dim+vb_dim+jb_dim+hla_dim, out_dim)
        
    def forward(self, 
                seq_tokens:torch.Tensor, # [B, L]
                embedding_single:torch.Tensor, # [B, L, 384]
                embedding_pair:torch.Tensor, # [B, L, L, 128]
                chain_encoding:torch.Tensor, # [B, L]
                distance_embedding:torch.Tensor, # [B, L, L, 256]
                pae_embedding:torch.Tensor, # [B, L, L, 16]
                plddts:torch.Tensor, # [B, L]
                va_token:torch.Tensor, 
                ja_token:torch.Tensor,
                vb_token:torch.Tensor,
                jb_token:torch.Tensor,
                hla_token:torch.Tensor,
                **kwargs):
        B, L = seq_tokens.shape
        seq_emb = self.aa_embedding(seq_tokens) # [B, L, 56]
        single_mask = (seq_tokens != 0).float()
        pair_mask = single_mask[:, None, :] * single_mask[:, :, None] # [B, L, L]
        single_emb = self.signle_compress(embedding_single) # [B,L,384]->[B, L, 64]
        chain_emb = self.chain_embedding(chain_encoding) # [B, L, 8]

        plddt_boundary = torch.linspace(0., 100., 20).to(plddts.device) # [1, 20]
        plddt_bins = torch.sum(plddts[..., None] > plddt_boundary, dim=-1) # [B, L]
        plddt_emb = self.plddt_embedding(plddt_bins) # [B, L, 16]

        pair_emb = self.pair_compress(embedding_pair) #[B,L,L,128]->[B, L, L, 64]
        distance_emb = self.distance_compress(distance_embedding) #[B, L, L, 256]->[B, L,L, 64]
        pae_emb = self.pae_compress(pae_embedding) # [B, L,L, 16]->[B, L, L, 16]

        va_emb = self.va_embedding(va_token) # [B, d]
        ja_emb = self.ja_embedding(ja_token) # [B, d]
        vb_emb = self.vb_embedding(vb_token) # [B, d]
        jb_emb = self.jb_embedding(jb_token)
        hla_emb = self.hla_embedding(hla_token)
        
        # position encoding
        positions = chain_encoding * 100 + torch.arange(L)[None, :].to(chain_encoding.device) # [B, L]
        positions_pair = positions[:, :, None] - positions[:, None, :] # [B, L, L]

        single_pos_emb = self.single_position_encoding(positions) #[B,L]->[B, L, 144]
        pair_pos_emb = self.pair_position_encoding(positions_pair) # [B, L, L] -> [B, L, L, 144]

        single_embedding = torch.cat([seq_emb, single_emb, chain_emb, plddt_emb], dim=-1) # [B, L, 144]
        pair_embedding = torch.cat([pair_emb, distance_emb, pae_emb], dim=-1) # [B, L, L, 144]
        point_embedding = torch.cat([va_emb, ja_emb, vb_emb, jb_emb, hla_emb], dim=-1) # [B, d+d+d+d+d]

        single_embedding = single_embedding + single_pos_emb # [B, L, 144]
        pair_embedding = pair_embedding + pair_pos_emb # [B, L, L, 144]

        single_embedding = self.single_out(single_embedding + single_pos_emb)
        pair_embedding = self.pair_out(pair_embedding + pair_pos_emb)
        point_embedding = self.point_out(torch.cat([va_emb, ja_emb, vb_emb, jb_emb, hla_emb], dim=-1)) # [B, d+d+d+d+d]

        return single_embedding, pair_embedding, point_embedding, single_mask, pair_mask #[B, L, 144], [B, L, L, 144], [B, d+d+d+d+d], [B, L], [B, L, L]