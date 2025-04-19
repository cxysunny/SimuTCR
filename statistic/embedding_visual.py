import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import os
from glob import glob
import time

# 设置随机种子确保可重复性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_embeddings(file_path):
    """加载和处理AlphaFold3嵌入向量文件"""
    data = np.load(file_path)
    
    # 获取单链嵌入和对嵌入
    single_emb = data['single_embeddings']  # [L, 384]
    pair_emb = data['pair_embeddings']      # [L, L, 128]
    
    # 1. 提取单链嵌入的平均值特征
    single_mean = np.mean(single_emb, axis=0)  # [384]
    
    # 2. 提取对嵌入的统计特征
    pair_mean = np.mean(pair_emb, axis=(0, 1))  # [128]
    
    # 3. 合并为一个特征向量
    features = np.concatenate([single_mean, pair_mean])  # [384+128=512]
    
    return features

def collect_embeddings(base_dir):
    """收集所有正负样本的嵌入向量"""
    # 收集正样本
    pos_pattern = os.path.join(base_dir, "pos_tcrpmhc", "*", "seed-1234_embeddings", "*_seed-1234_embeddings.npz")
    pos_files = glob(pos_pattern)
    
    # 收集负样本
    neg_pattern = os.path.join(base_dir, "neg_tcrpmhc", "*", "seed-1234_embeddings", "*_seed-1234_embeddings.npz")
    neg_files = glob(neg_pattern)
    
    print(f"找到{len(pos_files)}个正样本和{len(neg_files)}个负样本")
    
    # 限制样本数量，避免处理时间过长
    # max_samples = min(100, min(len(pos_files), len(neg_files)))
    # pos_files = pos_files[:max_samples]
    # neg_files = neg_files[:max_samples]
    
    # 加载嵌入向量
    pos_embeddings = [load_embeddings(f) for f in pos_files]
    neg_embeddings = [load_embeddings(f) for f in neg_files]
    
    # 创建标签
    pos_labels = np.ones(len(pos_embeddings))
    neg_labels = np.zeros(len(neg_embeddings))
    
    # 合并数据
    X = np.vstack(pos_embeddings + neg_embeddings)
    y = np.concatenate([pos_labels, neg_labels])
    
    return X, y

def visualize_embeddings(X, y):
    """使用t-SNE和UMAP可视化嵌入向量"""
    # 创建一个图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 设置颜色映射
    palette = {0: '#FF5733', 1: '#33A2FF'}  # 负样本红色，正样本蓝色
    
    # 1. t-SNE降维
    print("执行t-SNE降维...")
    start_time = time.time()
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE完成，耗时{time.time() - start_time:.2f}秒")
    
    # 绘制t-SNE散点图
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y, palette=palette,
        legend='full', alpha=0.8,
        ax=ax1
    )
    ax1.set_title('t-SNE Visualization of AF3 Embeddings')
    ax1.legend(title='Binding', labels=['Negative', 'Positive'])
    
    # 2. UMAP降维
    print("执行UMAP降维...")
    start_time = time.time()
    reducer = umap.UMAP(random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X)
    print(f"UMAP完成，耗时{time.time() - start_time:.2f}秒")
    
    # 绘制UMAP散点图
    sns.scatterplot(
        x=X_umap[:, 0], y=X_umap[:, 1],
        hue=y, palette=palette,
        legend='full', alpha=0.8,
        ax=ax2
    )
    ax2.set_title('UMAP Visualization of AF3 Embeddings')
    ax2.legend(title='Binding', labels=['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig('af3_embeddings_visualization.png', dpi=300)
    print("可视化图已保存为 'af3_embeddings_visualization.png'")
    plt.show()

if __name__ == "__main__":
    # 设置基本路径
    base_dir = "/home/xycui/project/af3_binding/dataset"
    
    # 收集嵌入向量
    print("正在收集嵌入向量...")
    X, y = collect_embeddings(base_dir)
    print(f"收集了{len(X)}个嵌入向量，特征维度为{X.shape[1]}")
    
    # 可视化嵌入向量
    visualize_embeddings(X, y)
    
    # 计算样本间距离统计
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # 计算组内距离和组间距离
    pos_distances = []
    neg_distances = []
    between_distances = []
    
    # 限制计算的样本对数量，避免计算过多
    max_pairs = 100
    
    # 随机选择一些样本对进行距离计算
    for _ in range(min(max_pairs, len(pos_indices))):
        i, j = np.random.choice(pos_indices, 2, replace=False)
        pos_distances.append(np.linalg.norm(X[i] - X[j]))
    
    for _ in range(min(max_pairs, len(neg_indices))):
        i, j = np.random.choice(neg_indices, 2, replace=False)
        neg_distances.append(np.linalg.norm(X[i] - X[j]))
    
    for _ in range(min(max_pairs, len(pos_indices) * len(neg_indices))):
        i = np.random.choice(pos_indices)
        j = np.random.choice(neg_indices)
        between_distances.append(np.linalg.norm(X[i] - X[j]))
    
    # 打印距离统计信息
    print("\n距离统计分析:")
    print(f"正样本内平均距离: {np.mean(pos_distances):.4f} ± {np.std(pos_distances):.4f}")
    print(f"负样本内平均距离: {np.mean(neg_distances):.4f} ± {np.std(neg_distances):.4f}")
    print(f"正负样本间平均距离: {np.mean(between_distances):.4f} ± {np.std(between_distances):.4f}")
