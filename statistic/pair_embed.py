import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from tqdm import tqdm

# # 限制线程数防止 OpenBLAS 报错
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
def extract_embedding(sample_path, mode='ab_pep'):
    try:
        sample = torch.load(sample_path, map_location='cpu')
        label = sample['label'].item()
        pair_embedding = sample['embedding_pair']  # [L, L, 128]
        chain_encoding = sample['chain_encoding'].numpy()  # [L]

        # 获取对应 index
        idx_a = np.where(chain_encoding == 1)[0]
        idx_b = np.where(chain_encoding == 2)[0]
        idx_p = np.where(chain_encoding == 3)[0]

        if mode == 'a_pep':
            consider = np.concatenate([idx_a, idx_p])
        elif mode == 'b_pep':
            consider = np.concatenate([idx_b, idx_p])
        elif mode == 'ab_pep':
            consider = np.concatenate([idx_a, idx_b, idx_p])
        else:
            raise ValueError("Invalid mode")

        # 提取并平均
        emb = pair_embedding[consider][:, consider, :].numpy()  # [L, L, D]
        emb_mean = np.mean(emb, axis=(0, 1))  # [D]
        return emb_mean, label

    except Exception as e:
        print(f"Error: {e} in {sample_path}")
        return None, None

def load_all_embeddings(root_dir, label, mode, max_samples=None):
    pattern = os.path.join(root_dir, "*", "seed-1234_embeddings", "*.pt")
    files = glob.glob(pattern)
    if max_samples:
        files = files[:max_samples]

    embeddings = []
    labels = []

    for f in tqdm(files, desc=f"{'Pos' if label else 'Neg'} - {mode}"):
        emb, lbl = extract_embedding(f, mode)
        if emb is not None:
            embeddings.append(emb)
            labels.append(label)
    return embeddings, labels

def plot_2d(X_tsne, X_umap, y, mode, output_dir):
    plt.figure(figsize=(10, 5))

    # TSNE
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], c='red', label='Negative', alpha=0.6)
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], c='blue', label='Positive', alpha=0.6)
    plt.title(f"t-SNE: {mode}")
    plt.legend()

    # UMAP
    plt.subplot(1, 2, 2)
    plt.scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], c='red', label='Negative', alpha=0.6)
    plt.scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], c='blue', label='Positive', alpha=0.6)
    plt.title(f"UMAP: {mode}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{mode}_embedding.png"), dpi=300)
    plt.close()

def run_analysis(pos_dir, neg_dir, mode, output_dir, max_samples=1000):
    pos_data, pos_labels = load_all_embeddings(pos_dir, 1, mode, max_samples)
    neg_data, neg_labels = load_all_embeddings(neg_dir, 0, mode, max_samples)

    X = np.array(pos_data + neg_data)
    y = np.array(pos_labels + neg_labels)

    if len(X) == 0:
        print(f"No data for mode: {mode}")
        return

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    X_tsne = tsne.fit_transform(X)

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X)

    # Plot
    plot_2d(X_tsne, X_umap, y, mode, output_dir)

def main():
    pos_dir = "./dataset/pos_tcrpmhc"
    neg_dir = "./dataset/neg_tcrpmhc"
    output_dir = "./pairembed"
    os.makedirs(output_dir, exist_ok=True)

    for mode in ['a_pep', 'b_pep', 'ab_pep']:
        print(f"\n--- Analyzing mode: {mode} ---")
        run_analysis(pos_dir, neg_dir, mode, output_dir)

if __name__ == "__main__":
    main()
