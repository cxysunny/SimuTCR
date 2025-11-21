import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# 从数据中提取信息
data_lines = [
    "THR 1 to ALA energy change is -1.48102",
    "ASP 2 to ALA energy change is -7.61099",
    "LEU 3 to ALA energy change is 1.16334",
    "GLY 4 to ALA energy change is 5.09091",
    "GLN 5 to ALA energy change is -13.378",
    "ASN 6 to ALA energy change is -0.364552",
    "LEU 7 to ALA energy change is 2.37977",
    "LEU 8 to ALA energy change is 1.56627",
    "TYR 9 to ALA energy change is 4.18752"
]

data_lines = [
    "ARG 1 to ALA energy change is 6.65898",
    "ALA 2 to ALA energy change is 0",
    "LYS 3 to ALA energy change is -0.581077",
    "PHE 4 to ALA energy change is 3.53996",
    "LYS 5 to ALA energy change is 2.87685",
    "GLN 6 to ALA energy change is 0.273774",
    "LEU 7 to ALA energy change is 1.43542",
    "LEU 8 to ALA energy change is 3.9758",
]

# 提取位置和能量值
positions = []
energy_changes = []
amino_acids = []

for line in data_lines:
    parts = line.split()
    amino_acids.append(parts[0])
    positions.append(int(parts[1]))
    energy_changes.append(float(parts[-1]))

# 创建图形
fig, ax = plt.subplots(figsize=(12, 6))

print(positions)
# 绘制折线图
line = ax.plot(positions, energy_changes, 
               color='#2E86AB', 
               marker='o', 
               markersize=8,
               linewidth=2.5,
               markerfacecolor='white',
               markeredgewidth=2,
               markeredgecolor='#2E86AB',
               label='Energy Change')

# 设置x轴和y轴标签
ax.set_xlabel('Residue Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Energy Change (kcal/mol)', fontsize=12, fontweight='bold')
ax.set_title('Energy Changes for Mutation to Alanine', fontsize=14, fontweight='bold', pad=20)

# 设置x轴刻度为整数位置
ax.set_xticks(positions)

# 添加零线参考
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--')

# 在数据点上方/下方添加数值标签
for i, (pos, energy, aa) in enumerate(zip(positions, energy_changes, amino_acids)):
    print(aa)
    # 确定标签位置（正值在上方，负值在下方）
    vertical_offset = 0.5 if energy >= 0 else -0.7
    
    # 添加能量值标签
    ax.annotate(f'{energy:.2f}', 
               xy=(pos, energy), 
               xytext=(0, vertical_offset),
               textcoords='offset points',
               ha='center',
               fontsize=9,
               fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 在x轴标签下方添加氨基酸名称
    ax.annotate(aa, 
               xy=(pos, 0), 
               xytext=(0, -10 if i % 2 == 0 else -40),  # 交替位置避免重叠
               textcoords='offset points',
               ha='center',
               fontsize=10,
               fontweight='bold',
               color='#333333')

# 设置y轴范围，留出足够的空间显示标签
y_min, y_max = min(energy_changes), max(energy_changes)
ax.set_ylim(y_min - 1.5, y_max + 1.5)

# 添加颜色区域表示能量变化的正负
ax.fill_between(positions, energy_changes, 0, 
                where=np.array(energy_changes) >= 0, 
                color='red', alpha=0.1, label='Unfavorable (ΔG > 0)')
ax.fill_between(positions, energy_changes, 0, 
                where=np.array(energy_changes) < 0, 
                color='green', alpha=0.1, label='Favorable (ΔG < 0)')

# 添加图例
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig('photo/alascan1.png', dpi=300)