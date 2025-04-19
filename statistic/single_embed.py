import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def load_embedding(file, label):
    """加载单个嵌入文件并返回均值特征向量和标签"""
    try:
        with np.load(file, mmap_mode='r') as npz:  # 使用内存映射加速大文件读取
            # emb = npz['pair_embeddings']
            # emb_mean = np.mean(emb, axis=(0,1))
            emb = npz['single_embeddings']
            emb_mean = np.mean(emb, axis=0)
        return emb_mean, label
    except Exception:
        return None, None

def load_all_embeddings(root_dir, label, max_samples=None):
    """并行加载目录下的所有嵌入文件"""
    pattern = os.path.join(root_dir, "*", "seed-1234_embeddings", "*_seed-1234_embeddings.npz")
    files = glob.glob(pattern)
    
    if max_samples and len(files) > max_samples:
        files = files[:max_samples]
    
    print(f"Processing {len(files)} files from {root_dir}")
    
    # 使用并行处理加速文件加载
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(partial(load_embedding, label=label), files),
            total=len(files),
            desc=f"Loading {'positive' if label==1 else 'negative'}"
        ))
    
    # 过滤掉加载失败的结果
    valid_results = [(data, lbl) for data, lbl in results if data is not None]
    if not valid_results:
        return [], []
    
    data_list, labels_list = zip(*valid_results)
    return list(data_list), list(labels_list)

def main():
    # 设置路径
    pos_dir = "./dataset/pos_tcrpmhc"
    neg_dir = "./dataset/neg_tcrpmhc"
    output_dir = "./pair_embed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 每类限制样本数加快处理 
    max_samples = 1500
    
    # 加载嵌入向量
    pos_data, pos_labels = load_all_embeddings(pos_dir, label=1, max_samples=max_samples)
    neg_data, neg_labels = load_all_embeddings(neg_dir, label=0, max_samples=max_samples)
    
    # 合并数据
    X = np.array(pos_data + neg_data)
    y = np.array(pos_labels + neg_labels)
    
    if len(X) == 0:
        print("No valid embeddings found!")
        return
    
    print(f"Analyzing {len(X)} samples ({len(pos_data)} positive, {len(neg_data)} negative)")
    
    # t-SNE 降维
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=min(30, len(X)//5),  # 根据样本量调整参数
        n_iter=500,
        method='barnes_hut',  # 使用更快的算法
        n_jobs=-1  # 并行计算
    )
    X_tsne = tsne.fit_transform(X)
    
    # UMAP 降维
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42,
        n_neighbors=min(15, len(X)//5),  # 根据样本量调整参数
        min_dist=0.1
    )
    X_umap = reducer.fit_transform(X)
    
    # 创建并保存图形
    plt.figure(figsize=(10, 5))
    
    # t-SNE 图
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='red', label='Negative', alpha=0.6, s=30)
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='blue', label='Positive', alpha=0.6, s=30)
    plt.title("t-SNE of AF3 Embeddings")
    plt.legend()
    
    # UMAP 图
    plt.subplot(1, 2, 2)
    plt.scatter(X_umap[y==0, 0], X_umap[y==0, 1], c='red', label='Negative', alpha=0.6, s=30)
    plt.scatter(X_umap[y==1, 0], X_umap[y==1, 1], c='blue', label='Positive', alpha=0.6, s=30)
    plt.title("UMAP of AF3 Embeddings")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'af3_embedding_visualization.png'), dpi=300)
    print(f"Visualization saved to {os.path.join(output_dir, 'af3_embedding_visualization.png')}")
    plt.show()

if __name__ == "__main__":
    main()
