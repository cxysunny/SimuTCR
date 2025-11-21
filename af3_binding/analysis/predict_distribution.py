import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame()
# all_results = pd.read_csv('/home/xycui/project/af3_binding/results/seq_only/epoch_30.csv',header=None)
# all_results = pd.read_csv('/home/xycui/project/af3_binding/results/4block_3lossweight/pepepoch20unseen.csv',header=None)
# all_results = pd.read_csv('/home/xycui/project/af3_binding/results/seq_plus_structure_9.23/epoch_9.csv',header=None)
# all_results = pd.read_csv('/home/xycui/project/af3_binding/results/seq_plus_structure_10.27/epoch_9.csv',header=None)
# all_results = pd.read_csv('/home/xycui/tcrbind/ROCunseen/tulipunseen.csv',header=None)
# all_results = pd.read_csv('/home/xycui/tcrbind/Randomunseen/epoch9ran.csv',header=None)
all_results = pd.read_csv('/home/xycui/tcrbind/Background/epoch_9bg.csv',header=None)
# all_results = pd.read_csv('/home/xycui/tcrbind/ROCunseen/epactunseen.csv',header=None)

positive_predictions = []
negative_predictions = []
for i in range(1,len(all_results)):
    if all_results.iloc[i,0] == '1':
        positive_predictions.append(float(all_results.iloc[i, 2]))
    else:
        negative_predictions.append(float(all_results.iloc[i, 2]))
   

# print(positive_predictions)
plt.figure(figsize=(8, 6))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# # 绘制直方图
# plt.hist(positive_predictions, bins=50, alpha=0.7, label='正样本', color='blue', density=True)
# plt.hist(negative_predictions, bins=50, alpha=0.7, label='负样本', color='red', density=True)

# plt.xlabel('预测值')
# plt.ylabel('密度')
# plt.title('正负样本预测分布')
# plt.legend()
# plt.grid(True, alpha=0.3)

data = pd.DataFrame({
    'predictions': np.concatenate([positive_predictions, negative_predictions]),
    'label': ['positive'] * len(positive_predictions) + ['negative'] * len(negative_predictions)
})

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='predictions', hue='label', 
             alpha=0.7, bins=50, kde=True, palette=['blue', 'red'])

plt.xlabel('score')
plt.ylabel('density')
plt.title('SimuTCR')
plt.grid(True, alpha=0.3)

plt.axvline(np.mean(positive_predictions), color='darkblue', linestyle='--', linewidth=2, 
            label=f'')
plt.axvline(np.mean(negative_predictions), color='darkred', linestyle='--', linewidth=2, 
            label=f'')

plt.savefig('photo/predict_distribution_SimuTCR_back.svg',format = 'svg', dpi=300)

# plt.clf()
# plt.figure(figsize=(10, 6))
# sns.kdeplot(positive_predictions, label='positive', color='blue', fill=True, alpha=0.5)
# sns.kdeplot(negative_predictions, label='negative', color='red', fill=True, alpha=0.5)

# plt.xlabel('score')
# plt.ylabel('density')
# plt.title('TULIP_KDE')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('photo/predict_distribution_TULIP_kde.svg', format = 'svg', dpi=300)