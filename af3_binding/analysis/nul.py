import numpy as np
import matplotlib.pyplot as plt

# 数据准备
categories = ['ROAUC', 'AUPR']
model_name = ['Panpep','ERGO2','TULIP','EPACT','OURmodel_stru','OURmodel_seq','OURmodel_full']
model_value =[[0.511,0.172],[0.521,0.179],[0.578,0.285],[0.480,0.183],[0.636,0.287],[0.577,0.196],[0.652,0.278]]
model_value = np.array(model_value)
color_list =  [
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
    '#ccebc5', '#ffed6f'
]
# 设置柱状图参数
bar_width = 0.2  # 柱宽
index = np.arange(len(categories))  # 分组索引
# 创建双柱状图
plt.figure(figsize=(10, 6))
ax1 = plt.subplot(111)
distance = 1.8
for i in range(len(categories)):
    if i == 1:
        ax2 = ax1.twinx()  # 创建第二个y轴
    for j in range(len(model_name)):
        if i == 0:
            # 绘制第一个模型的柱状图
            ax1.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j], color=color_list[j])
            ax1.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
        else:
            # 在第二个y轴上绘制其他模型的柱状图
            ax2.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j] if i==1 else '', color=color_list[j])
            ax2.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
        # # 绘制每个模型的柱状图
        # plt.bar((j) * bar_width+1.5*i, model_value[j,i], bar_width, label=model_name[j] if i==0 else '', color=color_list[j])
        # plt.text((j) * bar_width+1.5*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
    

# 添加标签和标题
plt.title('Unseen peptide:ROAUC & AUPR per Model', fontsize=15)
# plt.xlabel('test set', fontsize=12)
plt.ylabel('', fontsize=12)

# 设置 X 轴标签和刻度
plt.xlabel('', fontsize=12)
plt.xticks([0.5,2],['ROAUC', 'AUPR'], fontsize=12)

ax1.set_ylim((0.2,0.7))
ax2.set_ylim((0.0,0.5))
# plt.xticklabels(['ROAUC', 'AUPR'], fontsize=12)
# plt.xticks(index + bar_width/2, categories)  # 设置x轴刻度位置
plt.legend()  # 显示图例



# 调整布局
plt.tight_layout()
plt.savefig('af3_binding/photo/train.jpg',dpi=300)  # 保存图像