import numpy as np
import matplotlib.pyplot as plt

# # 数据准备
# categories = ['ROAUC', 'AUPR']
# model_name = ['Panpep','ERGO2','TULIP','EPACT','OURmodel_stru','OURmodel_seq','OURmodel_full']
# model_value =[[0.511,0.172],[0.521,0.179],[0.578,0.285],[0.480,0.183],[0.636,0.287],[0.598,0.223],[0.652,0.278]]
# model_value = np.array(model_value)
# color_list =  [
#     '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
#     '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
#     '#ccebc5', '#ffed6f'
# ]
# # 设置柱状图参数
# bar_width = 0.2  # 柱宽
# index = np.arange(len(categories))  # 分组索引
# # 创建双柱状图
# plt.figure(figsize=(10, 6))
# ax1 = plt.subplot(111)
# distance = 1.8
# for i in range(len(categories)):
#     if i == 1:
#         ax2 = ax1.twinx()  # 创建第二个y轴
#     for j in range(len(model_name)):
#         if i == 0:
#             # 绘制第一个模型的柱状图
#             ax1.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j], color=color_list[j])
#             ax1.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
#         else:
#             # 在第二个y轴上绘制其他模型的柱状图
#             ax2.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j] if i==1 else '', color=color_list[j])
#             ax2.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
#         # # 绘制每个模型的柱状图
#         # plt.bar((j) * bar_width+1.5*i, model_value[j,i], bar_width, label=model_name[j] if i==0 else '', color=color_list[j])
#         # plt.text((j) * bar_width+1.5*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
    

# # 添加标签和标题
# plt.title('Unseen peptide:ROAUC & AUPR per Model', fontsize=15)
# # plt.xlabel('test set', fontsize=12)
# plt.ylabel('', fontsize=12)

# # 设置 X 轴标签和刻度
# plt.xlabel('', fontsize=12)
# plt.xticks([0.5,2.5],['ROAUC', 'AUPR'], fontsize=12)

# ax1.set_ylim((0.2,0.7))
# ax2.set_ylim((0.0,0.5))
# # plt.xticklabels(['ROAUC', 'AUPR'], fontsize=12)
# # plt.xticks(index + bar_width/2, categories)  # 设置x轴刻度位置
# plt.legend()  # 显示图例



# # 调整布局
# plt.tight_layout()
# plt.savefig('af3_binding/photo/sales_comparison.jpg',dpi=300)  # 保存图像\

# plt.clf()
# 数据准备
# categories = ['ROAUC', 'AUPR']
# # model_name = ['Panpep','ERGO2','TULIP','EPACT','OURmodel_stru','OURmodel_seq','OURmodel_full']
# # model_value =[[0.506,0.174],[0.588,0.195],[0.681,0.421],[0.514,0.183],[0.675,0.370],[0.746,0.516],[0.772,0.557]]

# model_name = ['OURmodel_stru','OURmodel_seq','OURmodel_full']
# model_value =[[0.675,0.370],[0.746,0.516],[0.772,0.557]]
# model_value = np.array(model_value)
# #origin color
# # color_list =  [
# #     '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
# #     '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
# #     '#ccebc5', '#ffed6f'
# # ]
# #另一套
# # color_list = ['#91ccc0', '#7fabd1', '#f7ac53', '#ec6e66', '#b5ce4e', '#bd7795', '#7c7979',
# #               '#963b79', '#97d0c5', '#f39865', '#52aadc', '#c7c1de',
# #               '#eeb6d4', '#c89736', '#cd8875', '#d7584e']

# # 颜色（浅紫、浅蓝、浅绿、浅黄、浅橙、浅红）
# color_list = ['#E6E6FA', '#ADD8E6', '#90EE90', '#FFFACD', '#FFDAB9', '#FFB6C1']
# # 设置柱状图参数
# bar_width = 0.2  # 柱宽
# index = np.arange(len(categories))  # 分组索引
# # 创建双柱状图
# plt.figure(figsize=(10, 6))
# ax1 = plt.subplot(111)
# distance = 1.8
# for i in range(len(categories)):
#     if i == 1:
#         ax2 = ax1.twinx()  # 创建第二个y轴
#     for j in range(len(model_name)):
#         if i == 0:
#             # 绘制第一个模型的柱状图
#             ax1.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j], color=color_list[j])
#             ax1.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
#         else:
#             # 在第二个y轴上绘制其他模型的柱状图
#             ax2.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j] if i==1 else '', color=color_list[j])
#             ax2.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
#         # # 绘制每个模型的柱状图
#         # plt.bar((j) * bar_width+1.5*i, model_value[j,i], bar_width, label=model_name[j] if i==0 else '', color=color_list[j])
#         # plt.text((j) * bar_width+1.5*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
    

# # 添加标签和标题
# plt.title('Overlap peptide:ROAUC & AUPR per Model', fontsize=15)
# # plt.xlabel('test set', fontsize=12)
# plt.ylabel('', fontsize=12)

# # 设置 X 轴标签和刻度
# plt.xlabel('', fontsize=12)
# plt.xticks([0.5,2.5],['ROAUC', 'AUPR'], fontsize=12)

# ax1.set_ylim((0.3,0.9))
# ax2.set_ylim((0.0,0.7))
# # plt.xticklabels(['ROAUC', 'AUPR'], fontsize=12)
# # plt.xticks(index + bar_width/2, categories)  # 设置x轴刻度位置
# plt.legend()  # 显示图例



# # 调整布局
# plt.tight_layout()
# plt.savefig('photo/overlap_colored.jpg',dpi=300)  # 保存图像

categories = ['ROAUC', 'AUPR']
#overlap
# model_name = ['Panpep','ERGO2','TULIP','EPACT','OURmodel_stru','OURmodel_seq','OURmodel_full']
# model_value =[[0.506,0.174],[0.588,0.195],[0.681,0.421],[0.514,0.183],[0.675,0.370],[0.746,0.516],[0.772,0.557]]
#unseen
model_name = ['Stru_only','Seq_only','SimuTCR']
model_value =[[0.636,0.287],[0.598,0.223],[0.66,0.286]]
model_value = np.array(model_value)
#origin color
# color_list =  [
#     '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
#     '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
#     '#ccebc5', '#ffed6f'
# ]
#另一套
# color_list = ['#91ccc0', '#7fabd1', '#f7ac53', '#ec6e66', '#b5ce4e', '#bd7795', '#7c7979',
#               '#963b79', '#97d0c5', '#f39865', '#52aadc', '#c7c1de',
#               '#eeb6d4', '#c89736', '#cd8875', '#d7584e']

# 颜色（浅紫、浅蓝、浅绿、浅黄、浅橙、浅红）
color_list = ['#83F7B6','#F0F99C','#F75C88']
# 设置柱状图参数
bar_width = 0.2  # 柱宽
index = np.arange(len(categories))  # 分组索引
# 创建双柱状图
plt.figure(figsize=(10, 6))
ax1 = plt.subplot(111)
distance = 1.0
for i in range(len(categories)):
    if i == 1:
        ax2 = ax1.twinx()  # 创建第二个y轴
    for j in range(len(model_name)):
        if i == 0:
            # 绘制第一个模型的柱状图
            # breakpoint()
            ax1.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j], color=color_list[j],edgecolor='gray', linewidth=1.2)
            ax1.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
        else:
            # 在第二个y轴上绘制其他模型的柱状图
            ax2.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j] if i==1 else '', color=color_list[j],edgecolor='gray', linewidth=1.2)
            ax2.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
        # # 绘制每个模型的柱状图
        # plt.bar((j) * bar_width+1.5*i, model_value[j,i], bar_width, label=model_name[j] if i==0 else '', color=color_list[j])
        # plt.text((j) * bar_width+1.5*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
    

# 添加标签和标题
plt.title('Ablation on unseen peptide:AUROC & AUPR per Model', fontsize=15)
# plt.xlabel('test set', fontsize=12)
plt.ylabel('', fontsize=12)

# 设置 X 轴标签和刻度
plt.xlabel('', fontsize=12)
plt.xticks([0.2,1.2],['AUROC', 'AUPR'], fontsize=12)

ax1.set_ylim((0.3,0.7))
ax2.set_ylim((0.0,0.5))
# plt.xticklabels(['ROAUC', 'AUPR'], fontsize=12)
# plt.xticks(index + bar_width/2, categories)  # 设置x轴刻度位置
plt.legend()  # 显示图例

# 加粗坐标轴
for spine in ['left', 'bottom', 'right']:
    ax1.spines[spine].set_linewidth(1.5)
ax2.spines['right'].set_linewidth(1.5)


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# 调整布局
plt.tight_layout()
plt.savefig('photo/ablation_unseen.svg',format='svg',dpi=300)  # 保存图像



plt.clf()
categories = ['ROAUC', 'AUPR']
#overlap
# model_name = ['Panpep','ERGO2','TULIP','EPACT','OURmodel_stru','OURmodel_seq','OURmodel_full']
# model_value =[[0.506,0.174],[0.588,0.195],[0.681,0.421],[0.514,0.183],[0.675,0.370],[0.746,0.516],[0.772,0.557]]
#unseen
model_name = ['Stru_only','Seq_only','SimuTCR']
model_value =[[0.675,0.370],[0.746,0.516],[0.772,0.557]]
model_value = np.array(model_value)
#origin color
# color_list =  [
#     '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
#     '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
#     '#ccebc5', '#ffed6f'
# ]
#另一套
# color_list = ['#91ccc0', '#7fabd1', '#f7ac53', '#ec6e66', '#b5ce4e', '#bd7795', '#7c7979',
#               '#963b79', '#97d0c5', '#f39865', '#52aadc', '#c7c1de',
#               '#eeb6d4', '#c89736', '#cd8875', '#d7584e']

# 颜色（浅紫、浅蓝、浅绿、浅黄、浅橙、浅红）
color_list = ['#83F7B6','#F0F99C','#F75C88']
# 设置柱状图参数
bar_width = 0.2  # 柱宽
index = np.arange(len(categories))  # 分组索引
# 创建双柱状图
plt.figure(figsize=(10, 6))
ax1 = plt.subplot(111)
distance = 1.0
for i in range(len(categories)):
    if i == 1:
        ax2 = ax1.twinx()  # 创建第二个y轴
    for j in range(len(model_name)):
        if i == 0:
            # 绘制第一个模型的柱状图
            # breakpoint()
            ax1.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j], color=color_list[j],edgecolor='gray', linewidth=1.2)
            ax1.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
        else:
            # 在第二个y轴上绘制其他模型的柱状图
            ax2.bar((j) * bar_width+distance*i, model_value[j,i], bar_width, label=model_name[j] if i==1 else '', color=color_list[j],edgecolor='gray', linewidth=1.2)
            ax2.text((j) * bar_width+distance*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
        # # 绘制每个模型的柱状图
        # plt.bar((j) * bar_width+1.5*i, model_value[j,i], bar_width, label=model_name[j] if i==0 else '', color=color_list[j])
        # plt.text((j) * bar_width+1.5*i, model_value[j,i] , str(model_value[j,i]), ha='center', va='bottom', fontsize=10)
    

# 添加标签和标题
plt.title('Ablation on overlap peptide:AUROC & AUPR per Model', fontsize=15)
# plt.xlabel('test set', fontsize=12)
plt.ylabel('', fontsize=12)

# 设置 X 轴标签和刻度
plt.xlabel('', fontsize=12)
plt.xticks([0.2,1.2],['AUROC', 'AUPR'], fontsize=12)

ax1.set_ylim((0.35,0.85))
ax2.set_ylim((0.25,0.75))
# plt.xticklabels(['ROAUC', 'AUPR'], fontsize=12)
# plt.xticks(index + bar_width/2, categories)  # 设置x轴刻度位置
plt.legend()  # 显示图例

# 加粗坐标轴
for spine in ['left', 'bottom', 'right']:
    ax1.spines[spine].set_linewidth(1.5)
ax2.spines['right'].set_linewidth(1.5)


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# 调整布局
plt.tight_layout()
plt.savefig('photo/ablation_overlap.svg',format='svg',dpi=300)  # 保存图像