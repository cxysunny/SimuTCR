import torch
import torch.nn as nn
from .features import Features

import torch.nn.functional as F

class PairAttention(nn.Module):
    """简化版的残基对注意力机制，替代所有三角形操作"""
    def __init__(self, dim=144, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.linear_g = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, dim, bias=False)
        
    def forward(self, z, pair_mask):
        # z: [B, L, L, D], pair_mask: [B, L, L]
        B, L, _, D = z.shape
        z = self.norm(z)  # [B, L, L, D]
        
        # QKV投影
        qkv = self.qkv(z)  # [B, L, L, 3D]
        qkv = qkv.reshape(B, L, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # [3, B, H, L, L, d]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, L, d]
        
        # 将L×L维度展平为一个维度处理
        q = q.reshape(B, self.num_heads, L*L, self.head_dim)  # [B, H, L*L, d]
        k = k.reshape(B, self.num_heads, L*L, self.head_dim)  # [B, H, L*L, d]
        v = v.reshape(B, self.num_heads, L*L, self.head_dim)  # [B, H, L*L, d]
        
        # 计算注意力
        attn = (q @ k.transpose(-1, -2)) / (self.head_dim**0.5)  # [B, H, L*L, L*L]
        
        # 创建mask
        mask = pair_mask.reshape(B, 1, L*L)  # [B, 1, L*L]
        mask = mask.unsqueeze(2) * mask.unsqueeze(3)  # [B, 1, L*L, L*L]
        attn = attn.masked_fill(~mask.bool(), -1e9)
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).reshape(B, self.num_heads, L, L, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, L, L, D)
        
        # 门控机制 (保持与原始代码风格一致)
        g = torch.sigmoid(self.linear_g(z))  # [B, L, L, D]
        out = self.linear_out(out) * g  # [B, L, L, D]
        
        return out * pair_mask.unsqueeze(-1)  # [B, L, L, D]

class AttentionPairBias(nn.Module):
    """带Pair偏置的注意力 (Algorithm 24)"""
    def __init__(self, dim=144, num_heads=8):
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

class PairUpdate(nn.Module):
    """Pair特征更新模块 - 简化版"""
    def __init__(self, dim=144, num_heads=8):
        super().__init__()
        # 使用单一注意力模块替代所有三角形操作
        self.pair_attn = PairAttention(dim, num_heads)
        self.transition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Linear(4*dim, dim)
        )
        
    def forward(self, pair, pair_mask):
        # 单一注意力更新
        pair = pair + self.pair_attn(pair, pair_mask)
        # Transition层保持不变
        pair = pair + self.transition(pair) * pair_mask.unsqueeze(-1)
        return pair

class SingleUpdate(nn.Module):
    """Single特征更新模块"""
    def __init__(self, dim=144, num_heads=8):
        super().__init__()
        self.attn = AttentionPairBias(dim, num_heads)
        self.transition = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Linear(4*dim, dim)
        )
        
    def forward(self, single, pair, single_mask):
        # 带Pair偏置的注意力
        single = single + self.attn(single, pair, single_mask)
        # Transition层
        single = single + self.transition(single) * single_mask.unsqueeze(-1)
        return single

class PairformerBlock(nn.Module):
    """完整的Pairformer块"""
    def __init__(self, dim=144, num_heads=8):
        super().__init__()
        self.pair_update = PairUpdate(dim, num_heads)
        self.single_update = SingleUpdate(dim, num_heads)
        
    def forward(self, single, pair, single_mask, pair_mask):
        # 更新Pair特征
        pair = self.pair_update(pair, pair_mask)
        # 更新Single特征
        single = self.single_update(single, pair, single_mask)
        return single, pair

class PairformerStack(nn.Module):
    """完整的Pairformer堆栈"""
    def __init__(self, num_blocks=4, dim=144, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            PairformerBlock(dim, num_heads) for _ in range(num_blocks)
        ])
        
    def forward(self, single, pair, single_mask, pair_mask):
        for block in self.blocks:
            single, pair = block(single, pair, single_mask, pair_mask)
        return single, pair

class SimpleModel(nn.Module):
    def __init__(self, feature_config:dict, model_config:dict):
        super(SimpleModel, self).__init__()
        self.features = Features(**feature_config)
        
        # 确保num_heads参数存在
        if 'num_heads' not in model_config:
            model_config['num_heads'] = 8  # 默认头数
        
        # 默认减少块数，简化模型复杂度
        if 'num_blocks' not in model_config:
            model_config['num_blocks'] = 2 # 默认使用2个块
        
        self.pairformer = PairformerStack(**model_config)
        d = 144 + feature_config.get('va_dim', 32) * 5  # 假设va_dim默认为32
        self.head = nn.Linear(d, 1)
        
    def forward(self, inputs):
        single_embedding, pair_embedding, point_embedding, single_mask, pair_mask = self.features(**inputs)
        single, pair = self.pairformer(single_embedding, pair_embedding, single_mask, pair_mask)
        x = torch.cat([torch.mean(single, dim=1), point_embedding], dim=-1) # [B, 144+5*va_dim]
        x = self.head(x) #[B, 1]
        return x.squeeze(-1) #[B]