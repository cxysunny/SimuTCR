import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from scipy import stats

# 设置数据路径
pos_dir = "/home/xycui/project/af3_binding/dataset/pos_tcrpmhc"
neg_dir = "/home/xycui/project/af3_binding/dataset/neg_tcrpmhc"

def find_embedding_files(base_dir):
    """查找所有embeddings.pt文件"""
    pattern = os.path.join(base_dir, "*/*_embeddings/*_embeddings.pt")
    return glob(pattern)

def extract_chain_plddts(data):
    """从数据中提取每条链的pLDDT分数"""
    plddts = data['plddts']
    chain_encoding = data['chain_encoding']
    
    # 识别唯一的链编码
    unique_chains = torch.unique(chain_encoding).cpu().numpy()
    
    # 为每条链收集pLDDT
    chain_plddts = {}
    
    # 根据代码判断，chain_encoding为1,2,3分别对应CDR3α,CDR3β,peptide
    chain_names = {1: 'CDR3α', 2: 'CDR3β', 3: 'peptide'}
    
    for chain_id in unique_chains:
        chain_id = int(chain_id)
        if chain_id not in chain_names:
            continue
            
        # 提取该链的所有pLDDT值
        mask = (chain_encoding == chain_id)
        chain_plddt = plddts[mask].cpu().numpy()
        
        chain_plddts[chain_names[chain_id]] = chain_plddt
    
    # 整体pLDDT
    chain_plddts['Overall'] = plddts.cpu().numpy()
    
    # 添加TCR整体pLDDT (CDR3α + CDR3β)
    tcr_mask = (chain_encoding == 1) | (chain_encoding == 2)
    chain_plddts['TCR'] = plddts[tcr_mask].cpu().numpy()
    
    return chain_plddts

def analyze_plddts():
    """分析正负样本的pLDDT分布"""
    # 查找所有embeddings文件
    pos_files = find_embedding_files(pos_dir)
    neg_files = find_embedding_files(neg_dir)
    
    print(f"找到 {len(pos_files)} 个正样本和 {len(neg_files)} 个负样本")
    
    # 收集所有pLDDT
    pos_plddts = {'CDR3α': [], 'CDR3β': [], 'peptide': [], 'TCR': [], 'Overall': []}
    neg_plddts = {'CDR3α': [], 'CDR3β': [], 'peptide': [], 'TCR': [], 'Overall': []}
    
    # 收集每个样本的平均pLDDT (用于t-test)
    pos_avg_plddts = {'CDR3α': [], 'CDR3β': [], 'peptide': [], 'TCR': [], 'Overall': []}
    neg_avg_plddts = {'CDR3α': [], 'CDR3β': [], 'peptide': [], 'TCR': [], 'Overall': []}
    
    # 处理正样本
    for file in tqdm(pos_files, desc="处理正样本"):
        try:
            data = torch.load(file, map_location='cpu')
            chain_plddts = extract_chain_plddts(data)
            
            for chain, plddts in chain_plddts.items():
                pos_plddts[chain].extend(plddts)
                pos_avg_plddts[chain].append(np.mean(plddts))
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    # 处理负样本
    for file in tqdm(neg_files, desc="处理负样本"):
        try:
            data = torch.load(file, map_location='cpu')
            chain_plddts = extract_chain_plddts(data)
            
            for chain, plddts in chain_plddts.items():
                neg_plddts[chain].extend(plddts)
                neg_avg_plddts[chain].append(np.mean(plddts))
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    # 打印样本数量和残基数量
    print("\n样本统计:")
    print(f"正样本数量: {len(pos_files)}")
    print(f"负样本数量: {len(neg_files)}")
    
    for chain in pos_plddts.keys():
        print(f"{chain}: 正样本残基数 = {len(pos_plddts[chain])}, 负样本残基数 = {len(neg_plddts[chain])}")
    
    # 绘制pLDDT分布图
    chains = ['CDR3α', 'CDR3β', 'peptide', 'TCR', 'Overall']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, chain in enumerate(chains):
        ax = axes[i]
        
        # 绘制密度分布
        if len(pos_plddts[chain]) > 0 and len(neg_plddts[chain]) > 0:
            sns.kdeplot(pos_plddts[chain], color='red', label='Positive', fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(neg_plddts[chain], color='blue', label='Negative', fill=True, alpha=0.3, ax=ax)
            
            # 计算残基级别pLDDT平均值
            pos_mean = np.mean(pos_plddts[chain])
            neg_mean = np.mean(neg_plddts[chain])
            
            # t检验 (使用样本平均值，避免残基级别自相关)
            t_stat, p_val = stats.ttest_ind(pos_avg_plddts[chain], neg_avg_plddts[chain], equal_var=False)
            
            # 添加垂直线表示平均值
            ax.axvline(pos_mean, color='red', linestyle='--', 
                      label=f'Pos mean: {pos_mean:.2f}')
            ax.axvline(neg_mean, color='blue', linestyle='--',
                      label=f'Neg mean: {neg_mean:.2f}')
            
            # 标题包含差异和p值
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.set_title(f'{chain} pLDDT (Diff: {pos_mean-neg_mean:.2f}, p={p_val:.4f} {significance})', fontsize=14)
            ax.set_xlabel('pLDDT', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            
            # 设置x轴范围为0-100 (pLDDT的有效范围)
            ax.set_xlim(0, 100)
            
            ax.legend(fontsize=10)
        
    # 删除多余的子图
    if len(axes) > len(chains):
        for i in range(len(chains), len(axes)):
            fig.delaxes(axes[i])
    
    plt.suptitle('pLDDT Distribution: Positive vs Negative TCR-pMHC Complexes', fontsize=16)
    plt.tight_layout()
    plt.savefig('plddt_distribution.png', dpi=300)
    
    # 打印统计结果表格
    print("\n统计分析结果:")
    print("=" * 80)
    print(f"{'区域':<10} {'正样本均值':<10} {'负样本均值':<10} {'差异':<10} {'p值':<10} {'显著性':<10}")
    print("-" * 80)
    
    for chain in chains:
        pos_mean = np.mean(pos_plddts[chain])
        neg_mean = np.mean(neg_plddts[chain])
        diff = pos_mean - neg_mean
        t_stat, p_val = stats.ttest_ind(pos_avg_plddts[chain], neg_avg_plddts[chain], equal_var=False)
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        print(f"{chain:<10} {pos_mean:<10.2f} {neg_mean:<10.2f} {diff:<10.2f} {p_val:<10.4f} {significance:<10}")
    
    print("=" * 80)
    print("分析完成，图像已保存为 'plddt_distribution.png'")

if __name__ == "__main__":
    analyze_plddts()
