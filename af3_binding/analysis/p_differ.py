import numpy as np
import pandas as pd
df = pd.DataFrame()
df['model_a_prob'] = pd.read_csv('/home/xycui/project/af3_binding/results/seq_only/epoch_30.csv',header=None)[2][1:]
df['model_b_prob'] = pd.read_csv('/home/xycui/project/af3_binding/results/4block_3lossweight/pepepoch20unseen.csv',header=None)[2][1:]
df['model_c_prob'] = pd.read_csv('/home/xycui/project/af3_binding/results/seq_plus_structure_10.27/epoch_9.csv',header=None)[2][1:]
df['true_label'] = pd.read_csv('/home/xycui/project/af3_binding/results/seq_only/epoch_30.csv',header=None)[0][1:]

#overlap
# df['model_a_prob'] = pd.read_csv('/home/xycui/project/af3_binding/results/seq_only/overlap.csv',header=None)[2][1:]
# df['model_b_prob'] = pd.read_csv('/home/xycui/project/af3_binding/results/4block_3lossweight/overlapstru.csv',header=None)[2][1:]
# df['model_c_prob'] = pd.read_csv('/home/xycui/project/af3_binding/results/seq_plus_structure_10.27/overlap.csv',header=None)[2][1:]
# df['true_label'] = pd.read_csv('/home/xycui/project/af3_binding/results/seq_only/overlap.csv',header=None)[0][1:]



df = df.astype(float)

df['std'] = df[['model_a_prob', 'model_b_prob', 'model_c_prob']].std(axis=1)

df['conf_a'] = (df['model_a_prob'] - df['model_a_prob'].min())/(df['model_a_prob'].max() - df['model_a_prob'].min())
df['conf_b'] = (df['model_b_prob'] - df['model_b_prob'].min())/(df['model_b_prob'].max() - df['model_b_prob'].min())   
df['conf_c'] = (df['model_c_prob'] - df['model_c_prob'].min())/(df['model_c_prob'].max() - df['model_c_prob'].min())

high_confidence_threshold = 0.4  # |prob - 0.5| > 0.4
low_confidence_threshold = 0.2   # |prob - 0.5| < 0.2
high_std_threshold = 0         # 标准差阈值，可根据数据分布调整

# 筛选出高差异的样本
high_std_samples = df[df['std'] > high_std_threshold]

# 定义一个函数来检查我们想要的模式
def check_pattern(row):
    confs = [row['conf_a'], row['conf_b'], row['conf_c']]
    # 计算有多少个模型是高置信的
    num_high_conf = sum(1 for c in confs if c > high_confidence_threshold and confs[2] > high_confidence_threshold)
    # 计算有多少个模型是低置信的
    num_low_conf = sum(1 for c in confs if c < low_confidence_threshold)
    
    # 我们想要的模式：恰好只有一个模型高置信，且至少有一个模型低置信（或两个都低，或一个低一个相反）
    # 更简单的条件：只有一个模型高置信，其他两个不高
    return num_high_conf == 1 and num_low_conf > 1

# 定义一个函数来检查我们想要的模式
def check_high(row):
    confs = [row['conf_a'], row['conf_b'], row['conf_c']]
    # 计算有多少个模型是高置信的
    num_high_conf = 1 if confs[2] > high_confidence_threshold else 0
    # 计算有多少个模型是低置信的
    num_low_conf = sum(1 for c in confs if c < low_confidence_threshold)
    
    # 我们想要的模式：恰好只有一个模型高置信，且至少有一个模型低置信（或两个都低，或一个低一个相反）
    # 更简单的条件：只有一个模型高置信，其他两个不高
    return num_high_conf == 1 and num_low_conf > 1


# 定义一个函数来检查我们想要的模式
def check_low(row):
    confs = [row['conf_a'], row['conf_b'], row['conf_c']]
    # 计算有多少个模型是高置信的
    num_high_conf = 1 if confs[2] < low_confidence_threshold else 0
    # 计算有多少个模型是低置信的
    num_low_conf = sum(1 for c in confs if c > high_confidence_threshold)
    
    # 我们想要的模式：恰好只有一个模型高置信，且至少有一个模型低置信（或两个都低，或一个低一个相反）
    # 更简单的条件：只有一个模型高置信，其他两个不高
    return num_high_conf == 1 and num_low_conf > 1

def check_low_stru(row):
    confs = [row['conf_a'], row['conf_b'], row['conf_c']]
    # 计算有多少个模型是高置信的
    num_high_conf = 1 if confs[1] < low_confidence_threshold else 0
    # 计算有多少个模型是低置信的
    num_low_conf = sum(1 for c in confs if c > high_confidence_threshold)
    
    # 我们想要的模式：恰好只有一个模型高置信，且至少有一个模型低置信（或两个都低，或一个低一个相反）
    # 更简单的条件：只有一个模型高置信，其他两个不高
    return num_high_conf == 1 and num_low_conf > 1


def check_low_seq(row):
    confs = [row['conf_a'], row['conf_b'], row['conf_c']]
    # 计算有多少个模型是高置信的
    num_high_conf = 1 if confs[0] < low_confidence_threshold else 0
    # 计算有多少个模型是低置信的
    num_low_conf = sum(1 for c in confs if c > high_confidence_threshold)
    
    # 我们想要的模式：恰好只有一个模型高置信，且至少有一个模型低置信（或两个都低，或一个低一个相反）
    # 更简单的条件：只有一个模型高置信，其他两个不高
    return num_high_conf == 1 and num_low_conf > 1

# 应用函数
result_df = high_std_samples[high_std_samples.apply(check_high, axis=1)]
print(result_df)

result_df = high_std_samples[high_std_samples.apply(check_low, axis=1)]
print(result_df)

result_df = high_std_samples[high_std_samples.apply(check_low_stru, axis=1)]
print(result_df)

result_df = high_std_samples[high_std_samples.apply(check_low_seq, axis=1)]
print(result_df)