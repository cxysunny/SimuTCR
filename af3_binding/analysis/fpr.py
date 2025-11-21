import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 数据
models = ['Seq_only', 'Structure_only', 'Full']
metrics = ['FPR', 'FNR', 'TPR', 'TNR']

data = {
    'FPR': [0.09, 0.02, 0.14],
    'FNR': [0.72, 0.27, 0.38],
    'TPR': [0.16, 0.09, 0.27],
    'TNR': [0.81, 0.43, 0.60]
}

# 颜色方案
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
markers = ['o', 's', '^', 'D']

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制每条线
for i, metric in enumerate(metrics):
    ax.plot(models, data[metric], 
            label=metric, 
            color=colors[i], 
            marker=markers[i], 
            markersize=8,
            linewidth=2.5,
            markerfacecolor='white',
            markeredgewidth=2)

# 美化图形
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 设置y轴范围
ax.set_ylim(0, 0.9)

# 添加图例
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# 在数据点旁边添加数值标签
for i, metric in enumerate(metrics):
    for j, model in enumerate(models):
        ax.annotate(f'{data[metric][j]:.2f}', 
                   xy=(j, data[metric][j]), 
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   alpha=0.8)

# 调整布局
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
# 显示图形
plt.savefig('photo/tprfpr_10.27.svg',format = 'svg',)