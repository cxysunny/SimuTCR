# import torch
# import torch.nn as nn
# from .features import Features

# import torch.nn.functional as F

# class TriangleMultiplication(nn.Module):
#     """三角形乘法更新 (Algorithm 12 & 13)"""
#     def __init__(self, dim=144, c=128, outgoing=True):
#         super().__init__()
#         self.outgoing = outgoing #outgoing or incoming,沿着哪一方向进行三角形乘法更新
#         self.norm = nn.LayerNorm(dim)
#         self.norm1 = nn.LayerNorm(c)
#         self.linear_ab = nn.Linear(dim, 2*c, bias=False)
#         self.linear_abg = nn.Linear(dim, 2*c, bias=False)
#         self.linear_g = nn.Linear(dim, dim, bias=False)
#         self.linear_out = nn.Linear(c, dim, bias=False)
        
#     def forward(self, z, pair_mask):
#         # z: [B, L, L, D], pair_mask: [B, L, L]
#         B, L, _, D = z.shape
#         z = self.norm(z) # [B, L, L, D]
        
#         # 计算a和b
#         ab = torch.sigmoid(self.linear_abg(z))* self.linear_ab(z) # [B, L, L, 2C]
#         a, b = torch.chunk(ab, 2, dim=-1) # [B, L, L, C]
        
#         # 应用pair mask
#         a = a * pair_mask.unsqueeze(-1) # [B, L, L, C]
#         b = b * pair_mask.unsqueeze(-1)
        
#         # 计算门控
#         g = torch.sigmoid(self.linear_g(z)) # [B, L, L, D]
        
#         if self.outgoing:
#             a = a.permute(0, 3, 1, 2)  # [B, C, L, L]
#             b = b.permute(0, 3, 1, 2)
#             out = torch.einsum('bcik,bcjk->bcij', a, b) # [B, C, L, L]
#         else:
#             a = a.permute(0, 3, 1, 2)  # [B, C, L, L]
#             b = b.permute(0, 3, 1, 2)
#             out = torch.einsum('bcki,bckj->bcij', a, b) # [B, C, L, L]
            
#         out = out.permute(0, 2, 3, 1)  # [B, L, L, C]
#         out = out * pair_mask.unsqueeze(-1) # [B, L, L, C]
#         out = self.norm1(out)
#         out = self.linear_out(out) * g # [B, L, L, D]
#         return out * pair_mask.unsqueeze(-1) # [B, L, L, D]

# class TriangleAttention(nn.Module):
#     """三角形注意力 (Algorithm 14 & 15)"""
#     def __init__(self, dim=144, c=32, num_heads=4, starting_node=True):
#         super().__init__()
#         self.starting_node = starting_node
#         self.num_heads = num_heads
#         self.head_dim = c // num_heads 
        
#         self.norm = nn.LayerNorm(dim)
#         self.linear_qkv = nn.Linear(dim, 3*c, bias=False)
#         self.linear_b = nn.Linear(dim, num_heads, bias=False)
#         self.linear_g = nn.Linear(dim, dim, bias=False)
#         self.linear_out = nn.Linear(c, dim, bias=False)
        
#     def forward(self, z, pair_mask):
#         B, L, _, D = z.shape
#         z = self.norm(z) # [B, L, L, D]
        
#         # 投影QKV和偏置
#         qkv = self.linear_qkv(z)  # [B, L, L, 3C]
#         q, k, v = torch.chunk(qkv, 3, dim=-1) # [B, L, L, C]
        
#         # 分头处理
#         q = q.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4) # [B, H, L, L, d]
#         k = k.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
#         v = v.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        
#         attn_bias = self.linear_b(z).permute(0, 3, 1, 2)  # [B, H, L, L]
#         attn_mask = pair_mask.unsqueeze(1).unsqueeze(-1).float()  #[B,1,L,L,1]

#         if self.starting_node:
#             # Starting node: 固定i, j注意k
#             # 计算注意力分数: q[b,h,i,j,:] 点乘 k[b,h,i,k,:]
#             attn =  torch.einsum('bhijd,bhikd->bhijk', q, k) / (self.head_dim**0.5) # [B, H, I,J,K]
#             attn_mask = torch.einsum('bhijd,bhikd->bhijk', attn_mask, attn_mask).bool() # [B, H, L, L, L]
#             attn = attn + attn_bias[...,None,:,:] #[B,H,I,J,K]+[B,H,1,J,K]

#         else:
#             # Ending node: 固定j, i注意k
#             # 计算注意力分数: q[b,h,i,j,:] 点乘 k[b,h,k,j,:]
#             attn = torch.einsum('bhijd,bhkjd->bhikj', q, k) / (self.head_dim**0.5)  # [B, H, I, K, J]
#             attn_mask = torch.einsum('bhijd,bhkjd->bhijk', attn_mask, attn_mask).bool() # [B, H, L, L, L]
#             attn = attn.permute(0,1,3,4,2)+attn_bias[...,None,:,:] #[B,H,J,K,I]+[B,H,1,K,I]
#             attn = attn.permute(0,1,4,2,3) # [B, H, I, J, K]
        
#         attn = attn.masked_fill(~attn_mask, -1e9) # [B, H, L, L, L]
#         attn = F.softmax(attn, dim=-1) # [B, H, L, L, L]
        
#         # # 应用三角形mask
#         # if self.starting_node:
#         #     tri_mask = torch.ones(L, L, device=z.device).triu(diagonal=1).bool()
#         # else:
#         #     tri_mask = torch.ones(L, L, device=z.device).tril(diagonal=-1).bool()
#         # attn = attn.masked_fill(tri_mask.unsqueeze(0).unsqueeze(0), -1e9)
            
#         if self.starting_node:
#             out = torch.einsum('bhijk,bhikd->bhijd', attn, v) # [B, H, I, J, d]
#         else:
#             out = torch.einsum('bhijk,bhkjd->bhijd', attn, v) # [B, H, I, J, d]
#         out = out.permute(0, 2, 3, 1, 4).reshape(B, L, L, -1) # [B, L, L, C]          
        
#         # 应用门控
#         g = torch.sigmoid(self.linear_g(z)) # [B, L, L, D]
#         out = self.linear_out(out) * g # [B, L, L, D]
#         return out * pair_mask.unsqueeze(-1) # [B, L, L, D]

# class Dropout(nn.Module):
#     """Dropout层"""
#     def __init__(self, p, batch_dim):
#         """
#         Args:
#             p (float): Dropout rate
#             batch_dim: Dimension(s) along which the dropout mask is shared
#             """
#         super().__init__()
#         self.p = p
#         self.batch_dim = batch_dim
#         self.dropout = nn.Dropout(self.p)

#     def forward(self, x):
#         # x: [B, L, L, D]
#         shape=list(x.shape)
#         shape[self.batch_dim] = 1
#         mask = x.new_ones(shape)
#         mask = self.dropout(mask)
#         x = x * mask
#         return x # [B, L, L, D]

# class TransitionLayer(nn.Module):
#     def __init__(self, dim=128, n=4):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.linear_ab = nn.Linear(dim, n*dim*2)
#         self.linear_out = nn.Linear(n*dim, dim)

#     def forward(self, x):
#         x = self.norm(x)
#         ab = self.linear_ab(x)
#         a, b = torch.chunk(ab, 2, dim=-1)
#         x = self.linear_out(F.silu(a) * b)
#         return x

# class PairUpdate(nn.Module):
#     """Pair特征更新模块"""
#     def __init__(self, dim=144, num_heads=8):
#         super().__init__()
#         self.tri_mul_out = TriangleMultiplication(dim, outgoing=True)
#         self.tri_mul_in = TriangleMultiplication(dim, outgoing=False)
#         self.tri_attn_start = TriangleAttention(dim, starting_node=True)
#         self.tri_attn_end = TriangleAttention(dim, starting_node=False)
#         self.transition = TransitionLayer(dim,4)
#         # self.dropout_row = Dropout(0.25, -3) # [B, L, L, D]
#         # self.dropout_col = Dropout(0.25, -2) # [B, L, L, D]
        
#     def forward(self, pair, pair_mask):
#         # pair = pair + self.dropout_row(self.tri_mul_out(pair, pair_mask))
#         # pair = pair + self.dropout_row(self.tri_mul_in(pair, pair_mask))
#         # pair = pair + self.dropout_row(self.tri_attn_start(pair, pair_mask))
#         # pair = pair + self.dropout_col(self.tri_attn_end(pair, pair_mask))
#         pair = pair + self.tri_mul_out(pair, pair_mask)
#         pair = pair + self.tri_mul_in(pair, pair_mask)
#         pair = pair + self.tri_attn_start(pair, pair_mask)
#         pair = pair + self.tri_attn_end(pair, pair_mask)
        
#         pair = self.transition(pair) * pair_mask.unsqueeze(-1) #[B, L, L, D]
#         return pair

# class AttentionPairBias(nn.Module):
#     """带Pair偏置的注意力 (Algorithm 24)"""
#     def __init__(self, dim=144, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads 
        
#         self.norm = nn.LayerNorm(dim)
#         self.linear_q = nn.Linear(dim, dim, bias=False)
#         self.linear_kv = nn.Linear(dim, 2*dim, bias=False)
#         self.linear_b = nn.Linear(dim, num_heads, bias=False)
#         self.linear_out = nn.Linear(dim, dim, bias=False)
        
#     def forward(self, single, pair, single_mask):
#         B, L, D = single.shape
#         single = self.norm(single) # [B, L, D]
        
#         # 生成Q/K/V
#         q = self.linear_q(single).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) #[B, H, L, head_dim]
#         k, v = torch.chunk(self.linear_kv(single), 2, dim=-1)
#         k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
#         # 计算注意力分数
#         attn = torch.einsum('bhid,bhjd->bhij', q, k) / (self.head_dim**0.5) #[B,H,L,L]
#         attn_bias = self.linear_b(pair).permute(0, 3, 1, 2)  # [B, H, L, L]
#         attn = attn + attn_bias
        
#         # 应用mask
#         mask = single_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]
#         attn = attn.masked_fill(~mask, -1e9)
        
#         attn = F.softmax(attn, dim=-1)
#         out = torch.einsum('bhij,bhjd->bhid', attn, v) #[B, H, L, head_dim]
#         out = out.permute(0, 2, 1, 3).reshape(B, L, D) 
#         return self.linear_out(out) * single_mask.unsqueeze(-1) # [B, L, D]

# class SingleUpdate(nn.Module):
#     """Single特征更新模块"""
#     def __init__(self, dim=144, num_heads=8):
#         super().__init__()
#         self.attn = AttentionPairBias(dim)
#         self.transition = TransitionLayer(dim,4)
        
#     def forward(self, single, pair, single_mask):
#         # 带Pair偏置的注意力
#         single = single + self.attn(single, pair, single_mask)
#         # Transition层
#         single = single + self.transition(single) * single_mask.unsqueeze(-1)
#         return single

# class PairformerBlock(nn.Module):
#     """完整的Pairformer块"""
#     def __init__(self, dim=144, num_heads=8):
#         super().__init__()
#         self.pair_update = PairUpdate(dim, num_heads=num_heads)
#         self.single_update = SingleUpdate(dim, num_heads=num_heads)
        
#     def forward(self, single, pair, single_mask, pair_mask):
#         # 更新Pair特征
#         pair = self.pair_update(pair, pair_mask)
#         # 更新Single特征
#         single = self.single_update(single, pair, single_mask)
#         return single, pair

# class PairformerStack(nn.Module):
#     """完整的Pairformer堆栈"""
#     def __init__(self, num_blocks=4, dim=144, num_heads=8):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             PairformerBlock(dim, num_heads) for _ in range(num_blocks)
#         ])
        
#     def forward(self, single, pair, single_mask, pair_mask):
#         for block in self.blocks:
#             single, pair = block(single, pair, single_mask, pair_mask)
#         return single, pair


# class Model(nn.Module):
#     def __init__(self, feature_config:dict, model_config:dict, dim:int,use_seqfeat:bool=False, n_tgt_attn:int=1):
#         super(Model, self).__init__()
#         self.features = Features(**feature_config)
#         self.pairformer = PairformerStack(**model_config)
#         d = feature_config['out_dim']*2
#         self.head = nn.Linear(d, 1)
#         # self.head_1 = nn.Sequential(
#         #     nn.Linear(384, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 32),
#         #     nn.ReLU(),
#         #     nn.Linear(32, 1)
#         # )
        
#     def forward(self, inputs):
#         single_embedding, pair_embedding, point_embedding, single_mask, pair_mask = self.features(**inputs)
#         single, pair = self.pairformer(single_embedding, pair_embedding, single_mask, pair_mask)
#         x = torch.cat([torch.mean(single, dim=1), point_embedding], dim=-1) 
#         x = self.head(x) #[B, 1]
#         # x = self.head_1(torch.mean(inputs['embedding_single'], dim=1)) #[B, 384] -> [B, 1]

#         # for block in self.target_attn_blocks:
#         #     point_embedding = block(point_embedding, single, single_mask)
#         # x = self.head(point_embedding)
#         return x.squeeze(-1) #[B]

import torch
import torch.nn as nn
from .features import Features

import torch.nn.functional as F

class TriangleMultiplication(nn.Module):
    """三角形乘法更新 (Algorithm 12 & 13)"""
    def __init__(self, dim=144, c=128, outgoing=True):
        super().__init__()
        self.outgoing = outgoing #outgoing or incoming,沿着哪一方向进行三角形乘法更新
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(c)
        self.linear_ab = nn.Linear(dim, 2*c, bias=False)
        self.linear_abg = nn.Linear(dim, 2*c, bias=False)
        self.linear_g = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(c, dim, bias=False)
        
    def forward(self, z, pair_mask):
        # z: [B, L, L, D], pair_mask: [B, L, L]
        B, L, _, D = z.shape
        z = self.norm(z) # [B, L, L, D]
        
        # 计算a和b
        ab = torch.sigmoid(self.linear_abg(z))* self.linear_ab(z) # [B, L, L, 2C]
        a, b = torch.chunk(ab, 2, dim=-1) # [B, L, L, C]
        
        # 应用pair mask
        a = a * pair_mask.unsqueeze(-1) # [B, L, L, C]
        b = b * pair_mask.unsqueeze(-1)
        
        # 计算门控
        g = torch.sigmoid(self.linear_g(z)) # [B, L, L, D]
        
        if self.outgoing:
            a = a.permute(0, 3, 1, 2)  # [B, C, L, L]
            b = b.permute(0, 3, 1, 2)
            out = torch.einsum('bcik,bcjk->bcij', a, b) # [B, C, L, L]
        else:
            a = a.permute(0, 3, 1, 2)  # [B, C, L, L]
            b = b.permute(0, 3, 1, 2)
            out = torch.einsum('bcki,bckj->bcij', a, b) # [B, C, L, L]
            
        out = out.permute(0, 2, 3, 1)  # [B, L, L, C]
        out = out * pair_mask.unsqueeze(-1) # [B, L, L, C]
        out = self.norm1(out)
        out = self.linear_out(out) * g # [B, L, L, D]
        return out * pair_mask.unsqueeze(-1) # [B, L, L, D]

class TriangleAttention(nn.Module):
    """三角形注意力 (Algorithm 14 & 15)"""
    def __init__(self, dim=144, c=32, num_heads=4, starting_node=True):
        super().__init__()
        self.starting_node = starting_node
        self.num_heads = num_heads
        self.head_dim = c // num_heads 
        
        self.norm = nn.LayerNorm(dim)
        self.linear_qkv = nn.Linear(dim, 3*c, bias=False)
        self.linear_b = nn.Linear(dim, num_heads, bias=False)
        self.linear_g = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(c, dim, bias=False)
        
    def forward(self, z, pair_mask):
        B, L, _, D = z.shape
        z = self.norm(z) # [B, L, L, D]
        
        # 投影QKV和偏置
        qkv = self.linear_qkv(z)  # [B, L, L, 3C]
        q, k, v = torch.chunk(qkv, 3, dim=-1) # [B, L, L, C]
        
        # 分头处理
        q = q.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4) # [B, H, L, L, d]
        k = k.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        
        attn_bias = self.linear_b(z).permute(0, 3, 1, 2)  # [B, H, L, L]
        attn_mask = pair_mask.unsqueeze(1).unsqueeze(-1).float()  #[B,1,L,L,1]

        if self.starting_node:
            # Starting node: 固定i, j注意k
            # 计算注意力分数: q[b,h,i,j,:] 点乘 k[b,h,i,k,:]
            attn =  torch.einsum('bhijd,bhikd->bhijk', q, k) / (self.head_dim**0.5) # [B, H, I,J,K]
            attn_mask = torch.einsum('bhijd,bhikd->bhijk', attn_mask, attn_mask).bool() # [B, H, L, L, L]
            attn = attn + attn_bias[...,None,:,:] #[B,H,I,J,K]+[B,H,1,J,K]

        else:
            # Ending node: 固定j, i注意k
            # 计算注意力分数: q[b,h,i,j,:] 点乘 k[b,h,k,j,:]
            attn = torch.einsum('bhijd,bhkjd->bhikj', q, k) / (self.head_dim**0.5)  # [B, H, I, K, J]
            attn_mask = torch.einsum('bhijd,bhkjd->bhijk', attn_mask, attn_mask).bool() # [B, H, L, L, L]
            attn = attn.permute(0,1,3,4,2)+attn_bias[...,None,:,:] #[B,H,J,K,I]+[B,H,1,K,I]
            attn = attn.permute(0,1,4,2,3) # [B, H, I, J, K]
        
        attn = attn.masked_fill(~attn_mask, -1e9) # [B, H, L, L, L]
        attn = F.softmax(attn, dim=-1) # [B, H, L, L, L]
        
        # # 应用三角形mask
        # if self.starting_node:
        #     tri_mask = torch.ones(L, L, device=z.device).triu(diagonal=1).bool()
        # else:
        #     tri_mask = torch.ones(L, L, device=z.device).tril(diagonal=-1).bool()
        # attn = attn.masked_fill(tri_mask.unsqueeze(0).unsqueeze(0), -1e9)
            
        if self.starting_node:
            out = torch.einsum('bhijk,bhikd->bhijd', attn, v) # [B, H, I, J, d]
        else:
            out = torch.einsum('bhijk,bhkjd->bhijd', attn, v) # [B, H, I, J, d]
        out = out.permute(0, 2, 3, 1, 4).reshape(B, L, L, -1) # [B, L, L, C]          
        
        # 应用门控
        g = torch.sigmoid(self.linear_g(z)) # [B, L, L, D]
        out = self.linear_out(out) * g # [B, L, L, D]
        return out * pair_mask.unsqueeze(-1) # [B, L, L, D]

class Dropout(nn.Module):
    """Dropout层"""
    def __init__(self, p, batch_dim):
        """
        Args:
            p (float): Dropout rate
            batch_dim: Dimension(s) along which the dropout mask is shared
            """
        super().__init__()
        self.p = p
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.p)

    def forward(self, x):
        # x: [B, L, L, D]
        shape=list(x.shape)
        shape[self.batch_dim] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x # [B, L, L, D]

class TransitionLayer(nn.Module):
    def __init__(self, dim=128, n=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear_ab = nn.Linear(dim, n*dim*2)
        self.linear_out = nn.Linear(n*dim, dim)

    def forward(self, x):
        x = self.norm(x)
        ab = self.linear_ab(x)
        a, b = torch.chunk(ab, 2, dim=-1)
        x = self.linear_out(F.silu(a) * b)
        return x

class PairUpdate(nn.Module):
    """Pair特征更新模块"""
    def __init__(self, dim,dropout):
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(dim, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(dim, outgoing=False)
        self.tri_attn_start = TriangleAttention(dim, starting_node=True)
        self.tri_attn_end = TriangleAttention(dim, starting_node=False)
        self.transition = TransitionLayer(dim,4)
        # self.transition = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, 4*dim),
        #     nn.ReLU(),
        #     nn.Linear(4*dim, dim)
        # )
        self.dropout_row = Dropout(dropout, -3) # [B, L, L, D]
        self.dropout_col = Dropout(dropout, -2) # [B, L, L, D]
        
    def forward(self, pair, pair_mask):
        pair = pair + self.dropout_row(self.tri_mul_out(pair, pair_mask))
        pair = pair + self.dropout_row(self.tri_mul_in(pair, pair_mask))
        pair = pair + self.dropout_row(self.tri_attn_start(pair, pair_mask))
        pair = pair + self.dropout_col(self.tri_attn_end(pair, pair_mask))
        # pair = pair + self.tri_mul_out(pair, pair_mask)
        # pair = pair + self.tri_mul_in(pair, pair_mask)
        # pair = pair + self.tri_attn_start(pair, pair_mask)
        # pair = pair + self.tri_attn_end(pair, pair_mask)
        
        pair = self.transition(pair) * pair_mask.unsqueeze(-1) #[B, L, L, D]
        return pair

class AttentionPairBias(nn.Module):
    """带Pair偏置的注意力 (Algorithm 24)"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads 
        
        self.norm = nn.LayerNorm(dim)
        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_kv = nn.Linear(dim, 2*dim, bias=False)
        self.linear_b = nn.Linear(dim, num_heads, bias=False)
        self.linear_out = nn.Linear(dim, dim, bias=False)
        
    def forward(self, single, pair, single_mask):
        B, L, D = single.shape
        single = self.norm(single) # [B, L, D]
        
        # 生成Q/K/V
        q = self.linear_q(single).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) #[B, H, L, head_dim]
        k, v = torch.chunk(self.linear_kv(single), 2, dim=-1)
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        attn = torch.einsum('bhid,bhjd->bhij', q, k) / (self.head_dim**0.5) #[B,H,L,L]
        attn_bias = self.linear_b(pair).permute(0, 3, 1, 2)  # [B, H, L, L]
        attn = attn + attn_bias
        
        # 应用mask
        mask = single_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]
        attn = attn.masked_fill(~mask, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v) #[B, H, L, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, L, D) 
        return self.linear_out(out) * single_mask.unsqueeze(-1) # [B, L, D]

class SingleUpdate(nn.Module):
    """Single特征更新模块"""
    def __init__(self, dim,num_heads):
        super().__init__()
        self.attn = AttentionPairBias(dim,num_heads=num_heads)
        self.transition = TransitionLayer(dim,4)
        # self.transition = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, 4*dim),
        #     nn.ReLU(),
        #     nn.Linear(4*dim, dim)
        # )
        
    def forward(self, single, pair, single_mask):
        # 带Pair偏置的注意力
        single = single + self.attn(single, pair, single_mask)
        # Transition层
        single = single + self.transition(single) * single_mask.unsqueeze(-1)
        return single

class PairformerBlock(nn.Module):
    """完整的Pairformer块"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.pair_update = PairUpdate(dim,dropout=dropout)
        self.single_update = SingleUpdate(dim,num_heads=num_heads)
        
    def forward(self, single, pair, single_mask, pair_mask):
        # 更新Pair特征
        pair = self.pair_update(pair, pair_mask)
        # 更新Single特征
        single = self.single_update(single, pair, single_mask)
        return single, pair

class PairformerStack(nn.Module):
    """完整的Pairformer堆栈"""
    def __init__(self, num_blocks, dim, num_heads, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            PairformerBlock(dim=dim, num_heads=num_heads,dropout=dropout) for _ in range(num_blocks)
        ])
        
    def forward(self, single, pair, single_mask, pair_mask):
        for block in self.blocks:
            single, pair = block(single, pair, single_mask, pair_mask)
        return single, pair



class Model(nn.Module):
    def __init__(self, feature_config:dict, model_config:dict):
        super(Model, self).__init__()
        self.features = Features(**feature_config)
        self.pairformer = PairformerStack(**model_config)
        # d = 144 + feature_config['va_dim'] * 5
        d = feature_config['out_dim'] * 2
        self.head = nn.Linear(d, 1)
        # self.head_1 = nn.Sequential(
        #     nn.Linear(384, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        
    def forward(self, inputs):
        single_embedding, pair_embedding, point_embedding, single_mask, pair_mask = self.features(**inputs)
        single, pair = self.pairformer(single_embedding, pair_embedding, single_mask, pair_mask)
        x = torch.cat([torch.mean(single, dim=1), point_embedding], dim=-1) # [B, 144+5*va_dim]
        x = self.head(x) #[B, 1]
        # x = self.head_1(torch.mean(inputs['embedding_single'], dim=1)) #[B, 384] -> [B, 1]
        return x.squeeze(-1) #[B]

