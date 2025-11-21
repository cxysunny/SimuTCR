import os.path as osp
import os
import json
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

pos_dir = '/home/xycui/project/af3_binding/dataset/iedb_pos_v2_random500'
neg_dir = '/home/xycui/project/af3_binding/dataset/iedb_neg_v2_random1500'

pos_iptms = []
neg_iptms = []

i = 0
for sub_dir in os.listdir(pos_dir):
    if i % 1000 == 0:
        print(i)
    i += 1
    fl = osp.join(pos_dir, sub_dir, f'{sub_dir}_summary_confidences.json')
    with open(fl, 'r') as f:
        cfdata = json.load(f)
    pos_iptms.append(cfdata['chain_iptm'][-1])

for sub_dir in os.listdir(neg_dir):
    if i % 1000==0:
        print(i)
    i += 1
    fl = osp.join(neg_dir, sub_dir, f'{sub_dir}_summary_confidences.json')
    with open(fl, 'r') as f:
        cfdata = json.load(f)
    neg_iptms.append(cfdata['chain_iptm'][-1])

all_pred = np.concatenate([pos_iptms, neg_iptms])
all_label = np.concatenate([[1 for _ in pos_iptms], [0 for _ in neg_iptms]])
fpr, tpr, _ = metrics.roc_curve(all_label, all_pred)
auc = metrics.auc(fpr, tpr)
print(auc)

plt.hist(pos_iptms, bins=40, label='positive', density=True, alpha=0.6, color='red')
plt.hist(neg_iptms, bins=40, label='negative', density=True, alpha=0.6, color='blue')
plt.legend()
plt.savefig('./ptcr_iptms.png')