import os
import json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# 设置字体大小
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14
})

def collect_iptm_values(base_dir):
    iptm_values = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_summary_confidences.json"):
                json_path = os.path.join(root, file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    iptm = data.get("iptm")
                    if iptm is not None:
                        iptm_values.append(iptm)
    return iptm_values

# 目录路径
neg_dir = "/home/xycui/project/af3_binding/dataset/neg_tcrpmhc"
pos_dir = "/home/xycui/project/af3_binding/dataset/pos_tcrpmhc"
output_path = "/home/xycui/project/af3_binding/tcrpmhc_iptm.png"

# 提取 iptm 值
neg_iptms = collect_iptm_values(neg_dir)
pos_iptms = collect_iptm_values(pos_dir)

# 绘制归一化直方图（概率密度）
bins = 30
plt.figure(figsize=(10, 6))
plt.hist(neg_iptms, bins=bins, color='blue', alpha=0.6, label='Negative', edgecolor='black', density=True)
plt.hist(pos_iptms, bins=bins, color='red', alpha=0.6, label='Positive', edgecolor='black', density=True)
plt.xlabel("iptm", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.title("iptm Distribution (Normalized)", fontsize=18)
plt.legend()
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

# Mann-Whitney U 检验
stat, p_value = mannwhitneyu(pos_iptms, neg_iptms, alternative="two-sided")
print(f"Mann-Whitney U test: U={stat:.2f}, p-value={p_value:.4f}")
print(f"Figure saved to: {output_path}")


