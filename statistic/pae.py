import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
import seaborn as sns
from tqdm import tqdm

# Define data paths
pos_dir = "/home/xycui/project/af3_binding/dataset/pos_tcrpmhc"
neg_dir = "/home/xycui/project/af3_binding/dataset/neg_tcrpmhc"

def extract_chain_specific_pae(pae, chain_ids):
    """提取特定链间的PAE子矩阵
    
    Args:
        pae: PAE矩阵 [L, L]
        chain_ids: 链标识符列表
        
    Returns:
        字典，包含各种子矩阵的PAE统计值
    """
    # 创建链索引映射
    chain_indices = {}
    for i, chain in enumerate(chain_ids):
        if chain not in chain_indices:
            chain_indices[chain] = []
        chain_indices[chain].append(i)
    
    # 获取各链的索引
    A_indices = np.array(chain_indices.get('A', []))  # TCRα
    B_indices = np.array(chain_indices.get('B', []))  # TCRβ
    C_indices = np.array(chain_indices.get('C', []))  # peptide
    D_indices = np.array(chain_indices.get('D', []))  # MHC
    
    # 合并TCR索引 (A+B)
    TCR_indices = np.concatenate([A_indices, B_indices]) if len(A_indices) > 0 and len(B_indices) > 0 else np.array([])
    
    # 提取关键子矩阵
    pae_stats = {}
    
    # 1. TCR与peptide之间的PAE (最关键的相互作用)
    if len(TCR_indices) > 0 and len(C_indices) > 0:
        tcr_pep_pae = pae[np.ix_(TCR_indices, C_indices)]
        pae_stats['TCR_peptide'] = {
            'mean': np.mean(tcr_pep_pae),
            'median': np.median(tcr_pep_pae),
            'min': np.min(tcr_pep_pae),
            'max': np.max(tcr_pep_pae),
            'std': np.std(tcr_pep_pae)
        }
    
    # 2. TCR与MHC之间的PAE
    if len(TCR_indices) > 0 and len(D_indices) > 0:
        tcr_mhc_pae = pae[np.ix_(TCR_indices, D_indices)]
        pae_stats['TCR_MHC'] = {
            'mean': np.mean(tcr_mhc_pae),
            'median': np.median(tcr_mhc_pae),
            'min': np.min(tcr_mhc_pae),
            'max': np.max(tcr_mhc_pae),
            'std': np.std(tcr_mhc_pae)
        }
    
    # 3. peptide与MHC之间的PAE (应该在正负样本中相似)
    if len(C_indices) > 0 and len(D_indices) > 0:
        pep_mhc_pae = pae[np.ix_(C_indices, D_indices)]
        pae_stats['peptide_MHC'] = {
            'mean': np.mean(pep_mhc_pae),
            'median': np.median(pep_mhc_pae),
            'min': np.min(pep_mhc_pae),
            'max': np.max(pep_mhc_pae),
            'std': np.std(pep_mhc_pae)
        }
    
    # 4. 全局PAE (备用)
    pae_stats['global'] = {
        'mean': np.mean(pae),
        'median': np.median(pae),
        'min': np.min(pae),
        'max': np.max(pae),
        'std': np.std(pae)
    }
    
    return pae_stats

def load_and_process_file(file_path):
    """加载并处理单个PAE文件"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        pae = np.array(data['pae'])
        chain_ids = data['token_chain_ids']
        
        return extract_chain_specific_pae(pae, chain_ids)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def analyze_pae_distribution():
    """分析正负样本PAE分布差异"""
    # 查找所有置信度文件，排除带有_summary_的文件
    pos_files = glob(os.path.join(pos_dir, "*/*_confidences.json"))
    neg_files = glob(os.path.join(neg_dir, "*/*_confidences.json"))
    
    # 排除带有_summary_的文件
    pos_files = [f for f in pos_files if "_summary_" not in f]
    neg_files = [f for f in neg_files if "_summary_" not in f]
    
    print(f"Found {len(pos_files)} positive samples and {len(neg_files)} negative samples")
    
    # 保存处理结果
    pos_results = []
    neg_results = []
    
    # 处理正样本
    for file in tqdm(pos_files, desc="Processing positive samples"):
        result = load_and_process_file(file)
        if result:
            pos_results.append(result)
    
    # 处理负样本
    for file in tqdm(neg_files, desc="Processing negative samples"):
        result = load_and_process_file(file)
        if result:
            neg_results.append(result)
    
    # 聚合结果进行统计
    interaction_types = ['TCR_peptide', 'TCR_MHC', 'peptide_MHC', 'global']
    stats_types = ['mean', 'median', 'min', 'max', 'std']
    
    print("\n=== Chain-Specific PAE Analysis ===")
    
    for interaction in interaction_types:
        print(f"\n>> {interaction} Interaction:")
        
        # 收集数据进行比较
        pos_data = {}
        neg_data = {}
        
        for stat in stats_types:
            pos_values = [r[interaction][stat] for r in pos_results if interaction in r]
            neg_values = [r[interaction][stat] for r in neg_results if interaction in r]
            
            if len(pos_values) > 0 and len(neg_values) > 0:
                pos_data[stat] = pos_values
                neg_data[stat] = neg_values
                
                # 计算统计量
                pos_mean, pos_std = np.mean(pos_values), np.std(pos_values)
                neg_mean, neg_std = np.mean(neg_values), np.std(neg_values)
                
                # t检验
                t_stat, p_val = stats.ttest_ind(pos_values, neg_values, equal_var=False)
                
                print(f"  {stat.capitalize()}:")
                print(f"    Positive: {pos_mean:.4f} ± {pos_std:.4f}")
                print(f"    Negative: {neg_mean:.4f} ± {neg_std:.4f}")
                print(f"    Difference: {pos_mean - neg_mean:.4f}")
                print(f"    p-value: {p_val:.6f} ({'Significant' if p_val < 0.05 else 'Not significant'})")
        
        # 为每个交互类型创建一个可视化
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.size': 14})
        
        # 绘制均值的密度图
        sns.kdeplot(pos_data['mean'], color='red', label='Positive', fill=True, alpha=0.3)
        sns.kdeplot(neg_data['mean'], color='blue', label='Negative', fill=True, alpha=0.3)
        
        plt.title(f'PAE Distribution: {interaction.replace("_", "-")} Interface', fontsize=18)
        plt.xlabel('PAE Mean Value (Predicted Aligned Error Å)', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'pae_{interaction}_distribution.png', dpi=300)
    
    # 创建多子图比较
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({'font.size': 14})
    
    for i, interaction in enumerate(interaction_types):
        ax = axes[i//2, i%2]
        
        # 收集数据
        pos_means = [r[interaction]['mean'] for r in pos_results if interaction in r]
        neg_means = [r[interaction]['mean'] for r in neg_results if interaction in r]
        
        if len(pos_means) > 0 and len(neg_means) > 0:
            # 绘制密度图
            sns.kdeplot(pos_means, color='red', label='Positive', fill=True, alpha=0.3, ax=ax)
            sns.kdeplot(neg_means, color='blue', label='Negative', fill=True, alpha=0.3, ax=ax)
            
            ax.set_title(f'{interaction.replace("_", "-")} Interface', fontsize=16)
            ax.set_xlabel('PAE Mean Value (Å)', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.legend(fontsize=12)
    
    plt.suptitle('Comparison of PAE Distributions Across Different Interfaces', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pae_comparison.png', dpi=300)
    
    print("\nAnalysis complete. Visualization files saved.")

if __name__ == "__main__":
    analyze_pae_distribution()
