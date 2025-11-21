import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
file_path = "/home/xycui/project/af3_binding/immrep25_result.csv"
df = pd.read_csv(file_path)

# 创建图表
plt.figure(figsize=(12, 7))

# 创建颜色列表，结构化方法用浅蓝色，序列化方法用浅绿色
colors = ['lightblue' if s == 1 else 'lightgreen' for s in df['struc']]

# 绘制柱状图，柱子宽度缩小
bars = plt.bar(df['model'], df['AUC'], color=colors, width=0.8, alpha=0.8)

# 设置x轴标签旋转角度
plt.xticks(rotation=30, ha='right', fontsize=20)
plt.yticks(fontsize=20)
# 增大图例、坐标等文字大小
plt.title("Performance of Different Methods on Unseen Epitopes", fontsize=20)
plt.xlabel("Models", fontsize=20)
plt.ylabel("AUC", fontsize=20)

# 纵坐标不从0开始，设置合理的范围
plt.ylim(0.3, 0.6)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

#柱子上添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), ha='center', va='bottom', fontsize=17)

# 创建自定义图例，增大图例文字大小
import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='lightblue', label='Structure-based')
green_patch = mpatches.Patch(color='lightgreen', label='Sequence-based')
plt.legend(handles=[blue_patch, green_patch], loc='upper right', fontsize=20)

# 调整布局
plt.tight_layout()
plt.savefig("immrep25_result5.png", dpi=300)
# 显示图表
plt.show()