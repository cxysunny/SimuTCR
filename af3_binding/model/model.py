import torch
import torch.nn as nn
from .features import Features
from .hla_moe import MOE_MHC
import torch.nn.functional as F
from visualizer import get_local
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
        
    @get_local('attn_map')    
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
        # print(attn[0][0])       

        attn_bias = self.linear_b(pair).permute(0, 3, 1, 2)  # [B, H, L, L]
        attn = attn + attn_bias
        # print(attn_bias[0][0])
        # 应用mask
        mask = single_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]
        attn = attn.masked_fill(~mask, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        # print(attn.shape,attn[0][0])
        attn_map = attn
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
    def __init__(self, feature_config:dict, model_config:dict,hla_config:dict,mlm_config=None):
        super(Model, self).__init__()
        self.features = Features(**feature_config)
        self.pairformer = PairformerStack(**model_config)
        self.moe_mhc = MOE_MHC(hla_config)
        # d = 144 + feature_config['va_dim'] * 5
        d = feature_config['out_dim'] * 2
        self.head = nn.Linear(d, 1)

        self.pair_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU(),
            nn.Flatten(start_dim=-2, end_dim=-1),
        )
        # self.head_1 = nn.Sequential(
        #     nn.Linear(384, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        

        # MLM相关配置
        self.do_mlm = mlm_config is not None
        if self.do_mlm:
            self.mlm_head = nn.Linear(feature_config['out_dim'], feature_config['aa_size'])
            self.mlm_weight = mlm_config.get('weight', 0.1)

    @get_local('pair_emb')
    def forward(self, inputs,is_train=True):        
        # print(single_embedding.shape)
        seq_results,loss,seq_embedding = self.moe_mhc(**inputs)   
        
        # 添加MLM相关参数
        #9.10ver
        inputs_with_mlm = {
            **inputs,
            'seq_embed':seq_embedding, 
            'do_mlm': self.do_mlm,
            'training': is_train
        }
        # 6.23ver
        # inputs_with_mlm = {
        #     **inputs,
        #     'seq_embed':None, 
        #     'do_mlm': self.do_mlm,
        #     'training': is_train
        # }
        #
        seq_ = inputs['seq_tokens']
        cdr3a_ = inputs['cdr3a_tokens']
        cdr3b_ = inputs['cdr3b_tokens']
        peptide_ = inputs['peptide_tokens']
        # print(seq_,cdr3a_,cdr3b_,peptide_)


        outputs = self.features(**inputs_with_mlm)
        
        if self.do_mlm:
            single_embedding, pair_embedding, point_embedding, single_mask, pair_mask, orig_seq_tokens, mask_indices, mlm_labels = outputs
        else:
            single_embedding, pair_embedding, point_embedding, single_mask, pair_mask = outputs
            orig_seq_tokens, mask_indices,mlm_labels = None, None, None
        
        # print(single_embedding.shape, seq_embedding.shape)
        
        # 6.23ver
        # single_embedding = single_embedding + seq_embedding
        # 9.10ver
        #None 

        # single_embedding = torch.cat([single_embedding, seq_embedding], dim=1)  # [B, L+76, 128]

        # 6.27ver
        # single_embedding[:,:,56:56+64] = single_embedding[:,:,56:56+64] + seq_embedding
        # print(single_embedding.shape)
        single, pair = self.pairformer(single_embedding, pair_embedding, single_mask, pair_mask)
 
        # 主任务：结合预测
        x = torch.cat([torch.mean(single, dim=1), point_embedding], dim=-1) 

        binding_pred = self.head(x).squeeze(-1)  # [B]
        
        # print(pair.shape)
     

        pair_pred = pair            
        # print(pair_pred)
        # 8.20 ver
        pair_pred = self.pair_head(pair_pred).squeeze(-1)        
        # print(pair_pred)
        pair_emb = pair_pred.clone()

        # print(pair_pred.shape)
        pair_pred = torch.mean(pair_pred,dim = -1)
        # print(pair_pred.shape)
        pair_pred = torch.mean(pair_pred,dim = -1)
        # print(pair_pred.shape)        
        # 8.24ver
        # pair_pred = self.pair_head(pair_pred).squeeze(-1)
        # print(pair_pred.shape)
        # print(pair_pred.shape)
        # print(pair_pred.shape)
        # print(single.shape,pair.shape,torch.mean(torch.mean(pair,dim = 0),dim = -1))/
        # bind_max = torch.max(binding_pred, dim=0).values
        # bind_min = torch.min(binding_pred, dim=0).values
        # seq_max = torch.max(seq_results, dim=0).values
        # seq_min = torch.min(seq_results, dim=0).values
        # binding_pred = (binding_pred - bind_min) / (bind_max - bind_min + 1e-6)
        # seq_results = (seq_results - seq_min) / (seq_max - seq_min + 1e-6)
        # 
        # binding_pred = binding_pred + seq_results
        # binding_pred = seq_results
        binding_pred = binding_pred + seq_results# + pair_pred

        # binding_pred = seq_results.unsqueeze(0)
        
        result = {'binding_pred': binding_pred}

        result.update({
            'sup_loss': loss, 
        })        


        # MLM辅助任务
        if self.do_mlm:
            mlm_logits = self.mlm_head(single)  # [B, L, aa_size]
            if is_train:
                result.update({
                    'mlm_logits': mlm_logits, 
                    'mlm_labels': mlm_labels,
                    'mask_indices': mask_indices
                })
            else:
                 result.update({
                    'mlm_logits': mlm_logits, 
                    'orig_tokens': orig_seq_tokens,
                    'mask_indices': mask_indices
                })    
            
        return result

