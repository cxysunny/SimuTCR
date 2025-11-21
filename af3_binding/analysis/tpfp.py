import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
# 生成示例数据
np.random.seed(42)


# all_results = pd.read_csv('/home/xycui/project/af3_binding/results/seq_only/epoch_30.csv',header=None)
# all_results = pd.read_csv('/home/xycui/project/af3_binding/results/4block_3lossweight/pepepoch20unseen.csv',header=None)
# all_results = pd.read_csv('/home/xycui/project/af3_binding/results/seq_plus_structure_9.23/epoch_9.csv',header=None)
all_results = pd.read_csv('/home/xycui/project/af3_binding/results/seq_plus_structure_10.27/epoch_9.csv',header=None)
positive_predictions = []
negative_predictions = []
for i in range(1,len(all_results)):
    if all_results.iloc[i,0] == '1':
        positive_predictions.append(float(all_results.iloc[i, 2]))
    else:
        negative_predictions.append(float(all_results.iloc[i, 2]))

positive_predictions = np.array(positive_predictions)  # 正样本预测值
negative_predictions = np.array(negative_predictions)  # 负样本预测值

# 设置双阈值
positive_threshold = 0.4  # 真阳性阈值
negative_threshold = 0.2  # 真阴性阈值

# 真实标签
y_true_positive = np.ones(1000)  # 正样本的真实标签为1
y_true_negative = np.zeros(1000)  # 负样本的真实标签为0

# 根据双阈值进行分类
def classify_with_dual_threshold(predictions, pos_thresh, neg_thresh):
    """
    使用双阈值进行分类
    - 预测值 >= pos_thresh: 分类为正样本 (1)
    - 预测值 <= neg_thresh: 分类为负样本 (0)
    - 其他: 不确定 (-1)
    """
    results = np.zeros_like(predictions)
    results[predictions >= pos_thresh] = 1  # 正样本
    results[predictions <= neg_thresh] = 0  # 负样本
    results[(predictions > neg_thresh) & (predictions < pos_thresh)] = -1  # 不确定
    return results

# 对正负样本进行分类
y_pred_positive = classify_with_dual_threshold(positive_predictions, positive_threshold, negative_threshold)
y_pred_negative = classify_with_dual_threshold(negative_predictions, positive_threshold, negative_threshold)

# 合并所有样本
y_true = np.concatenate([y_true_positive, y_true_negative])
y_pred = np.concatenate([y_pred_positive, y_pred_negative])

# 统计分类结果
positive_certain_positive = np.sum(y_pred_positive == 1)
positive_certain_negative = np.sum(y_pred_positive == 0)
positive_uncertain = np.sum(y_pred_positive == -1)

negative_certain_positive = np.sum(y_pred_negative == 1)
negative_certain_negative = np.sum(y_pred_negative == 0)
negative_uncertain = np.sum(y_pred_negative == -1)

print("="*60)
print("双阈值分类结果分析")
print("="*60)
print(f"正样本阈值: {positive_threshold}, 负样本阈值: {negative_threshold}")
print("="*60)
print("正样本分类结果:")
print(f"  明确为正: {positive_certain_positive} ({positive_certain_positive/len(positive_predictions):.2%})")
print(f"  明确为负: {positive_certain_negative} ({positive_certain_negative/len(positive_predictions):.2%})")
print(f"  不确定: {positive_uncertain} ({positive_uncertain/len(positive_predictions):.2%})")
print()
print("负样本分类结果:")
print(f"  明确为正: {negative_certain_positive} ({negative_certain_positive/len(negative_predictions):.2%})")
print(f"  明确为负: {negative_certain_negative} ({negative_certain_negative/len(negative_predictions):.2%})")
print(f"  不确定: {negative_uncertain} ({negative_uncertain/len(negative_predictions):.2%})")
print("="*60)