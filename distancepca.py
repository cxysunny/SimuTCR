# import torch
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import glob
# from tqdm import tqdm
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Define paths
# neg_dir = '/home/xycui/project/af3_binding/dataset/neg_tcrpmhc_pt'
# pos_dir = '/home/xycui/project/af3_binding/dataset/pos_tcrpmhc_pt'

# def collect_distance_embeddings(root_dir, limit=5000):
#     """Collect distance embeddings from .pt files"""
#     all_embeddings = []
#     all_files = []
    
#     # Find all .pt files in the directory (recursively)
#     for pt_file in tqdm(glob.glob(f"{root_dir}/**/*.pt", recursive=True)[:limit]):
#         all_files.append(pt_file)
#         try:
#             # Load the pytorch file
#             data = torch.load(pt_file)
#             # Extract the distance embedding
#             dist_emb = data['distance_embedding']  # [L, L, 256]
#             all_embeddings.append(dist_emb)
#         except Exception as e:
#             print(f"Error loading {pt_file}: {e}")
    
#     return all_embeddings, all_files

# # Collect data
# print("Loading negative samples...")
# neg_embeddings, neg_files = collect_distance_embeddings(neg_dir)
# print(f"Loaded {len(neg_embeddings)} negative samples")

# print("Loading positive samples...")
# pos_embeddings, pos_files = collect_distance_embeddings(pos_dir)
# print(f"Loaded {len(pos_embeddings)} positive samples")

# # Feature extraction for clustering
# def extract_features_for_clustering(embeddings):
#     """Extract features from distance embeddings for clustering"""
#     features = []
    
#     for emb in embeddings:
#         # Flatten the embedding and compute statistics
#         flat_emb = emb.reshape(-1).numpy()
        
#         # Extract basic statistics
#         stats = [
#             np.mean(flat_emb),
#             np.std(flat_emb),
#             np.median(flat_emb),
#             np.max(flat_emb),
#             np.min(flat_emb)
#         ]
        
#         # Extract CDR3-peptide interface statistics
#         L = emb.shape[0]
#         peptide_start = 2*L//3  # Adjust as needed
#         cdr_pep_region = emb[:peptide_start, peptide_start:].numpy()
#         interface_stats = [
#             np.mean(cdr_pep_region),
#             np.std(cdr_pep_region),
#             np.median(cdr_pep_region)
#         ]
        
#         # Combine all features
#         feature_vector = stats + interface_stats
#         features.append(feature_vector)
    
#     return np.array(features)

# # Extract features
# neg_features = extract_features_for_clustering(neg_embeddings)
# pos_features = extract_features_for_clustering(pos_embeddings)

# # Combine all features and create labels
# all_features = np.vstack((neg_features, pos_features))
# labels = np.array([0] * len(neg_features) + [1] * len(pos_features))

# # Standardize features
# scaler = StandardScaler()
# all_features_scaled = scaler.fit_transform(all_features)

# # Apply dimensionality reduction
# print("Applying PCA...")
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(all_features_scaled)

# print("Applying t-SNE...")
# tsne = TSNE(n_components=2, perplexity=min(30, len(all_features_scaled)-1), random_state=42)
# tsne_result = tsne.fit_transform(all_features_scaled)

# # Apply clustering
# print("Performing K-means clustering...")
# kmeans = KMeans(n_clusters=2, random_state=42)
# cluster_labels = kmeans.fit_predict(all_features_scaled)

# # Create visualization
# plt.figure(figsize=(20, 10))

# # 1. PCA visualization colored by true labels
# plt.subplot(2, 2, 1)
# scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
# plt.title('PCA Projection (Colored by True Labels)')
# plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
# plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
# plt.colorbar(scatter, label='Label (0=Negative, 1=Positive)')

# # 2. PCA visualization colored by clusters
# plt.subplot(2, 2, 2)
# scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
# plt.title('PCA Projection (Colored by Clusters)')
# plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
# plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
# plt.colorbar(scatter, label='Cluster')

# # 3. t-SNE visualization colored by true labels
# plt.subplot(2, 2, 3)
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
# plt.title('t-SNE Projection (Colored by True Labels)')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.colorbar(scatter, label='Label (0=Negative, 1=Positive)')

# # 4. t-SNE visualization colored by clusters
# plt.subplot(2, 2, 4)
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
# plt.title('t-SNE Projection (Colored by Clusters)')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.colorbar(scatter, label='Cluster')

# plt.tight_layout()
# plt.savefig('distance_embedding_clustering.png', dpi=300)
# plt.show()

# # Calculate clustering metrics
# from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score

# # Compute confusion matrix
# cm = confusion_matrix(labels, cluster_labels)
# print("\nConfusion Matrix (may need label permutation):")
# print(cm)

# # Compute clustering metrics (invariant to label permutation)
# ari = adjusted_rand_score(labels, cluster_labels)
# nmi = normalized_mutual_info_score(labels, cluster_labels)

# print(f"\nClustering Evaluation:")
# print(f"Adjusted Rand Index: {ari:.4f}")
# print(f"Normalized Mutual Information: {nmi:.4f}")

# # Feature importance analysis
# plt.figure(figsize=(12, 6))
# feature_names = [
#     'Mean', 'Std Dev', 'Median', 'Max', 'Min',
#     'Interface Mean', 'Interface Std Dev', 'Interface Median'
# ]

# # PCA loadings for feature importance
# loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# plt.bar(feature_names, loadings[:, 0], alpha=0.7, label='PC1')
# plt.bar(feature_names, loadings[:, 1], alpha=0.7, label='PC2')
# plt.title('Feature Importance in PCA Components')
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300)
# plt.show()

# # 3D visualization if needed
# from mpl_toolkits.mplot3d import Axes3D

# # Apply 3D PCA
# pca_3d = PCA(n_components=3)
# pca_result_3d = pca_3d.fit_transform(all_features_scaled)

# fig = plt.figure(figsize=(15, 6))

# # 3D PCA with true labels
# ax1 = fig.add_subplot(121, projection='3d')
# scatter = ax1.scatter(
#     pca_result_3d[:, 0], 
#     pca_result_3d[:, 1], 
#     pca_result_3d[:, 2],
#     c=labels, 
#     cmap='coolwarm', 
#     alpha=0.7
# )
# ax1.set_title('3D PCA (True Labels)')
# ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
# ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
# ax1.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
# fig.colorbar(scatter, ax=ax1, label='Label')

# # 3D PCA with cluster labels
# ax2 = fig.add_subplot(122, projection='3d')
# scatter = ax2.scatter(
#     pca_result_3d[:, 0], 
#     pca_result_3d[:, 1], 
#     pca_result_3d[:, 2],
#     c=cluster_labels, 
#     cmap='viridis', 
#     alpha=0.7
# )
# ax2.set_title('3D PCA (Clusters)')
# ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
# ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
# ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
# fig.colorbar(scatter, ax=ax2, label='Cluster')

# plt.tight_layout()
# plt.savefig('3d_clustering.png', dpi=300)
# plt.show()

import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Define paths
neg_dir = '/home/xycui/project/af3_binding/dataset/neg_tcrpmhc_pt'
pos_dir = '/home/xycui/project/af3_binding/dataset/pos_tcrpmhc_pt'

def collect_distance_embeddings(root_dir, limit=1000):
    """Collect distance_embedding from .pt files"""
    all_embeddings = []
    all_files = []
    
    for pt_file in tqdm(glob.glob(f"{root_dir}/**/*.pt", recursive=True)[:limit]):
        all_files.append(pt_file)
        try:
            data = torch.load(pt_file)
            dist_emb = data['distance_embedding']  # [L, L, 256]
            all_embeddings.append(dist_emb)
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
    
    return all_embeddings, all_files

# Collect data
print("Loading negative samples...")
neg_embeddings, neg_files = collect_distance_embeddings(neg_dir)
print(f"Loaded {len(neg_embeddings)} negative samples")

print("Loading positive samples...")
pos_embeddings, pos_files = collect_distance_embeddings(pos_dir)
print(f"Loaded {len(pos_embeddings)} positive samples")

# Extract features for clustering
def extract_features(embeddings):
    """Extract features from distance_embedding for clustering"""
    features = []
    
    for emb in embeddings:
        # Extract global statistics
        flat_emb = emb.reshape(-1).numpy()
        stats = [
            np.mean(flat_emb),
            np.std(flat_emb),
            np.median(flat_emb),
            np.percentile(flat_emb, 25),
            np.percentile(flat_emb, 75)
        ]
        
        # Extract CDR3-peptide interface features
        L = emb.shape[0]
        cdr3a_end = L//3
        cdr3b_end = 2*L//3
        peptide_start = 2*L//3
        
        # CDR3a-peptide interface
        cdr3a_pep = emb[:cdr3a_end, peptide_start:].numpy()
        # CDR3b-peptide interface
        cdr3b_pep = emb[cdr3a_end:cdr3b_end, peptide_start:].numpy()
        
        interface_stats = [
            np.mean(cdr3a_pep),
            np.mean(cdr3b_pep),
            np.std(cdr3a_pep),
            np.std(cdr3b_pep)
        ]
        
        # Combine all features
        feature_vector = stats + interface_stats
        features.append(feature_vector)
    
    return np.array(features)

# Extract features
neg_features = extract_features(neg_embeddings)
pos_features = extract_features(pos_embeddings)

# Combine features and create labels
all_features = np.vstack((neg_features, pos_features))
labels = np.array([0] * len(neg_features) + [1] * len(pos_features))

# Standardize features
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

# Apply PCA dimensionality reduction
print("Applying PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_features_scaled)
var_explained = pca.explained_variance_ratio_

# Apply t-SNE dimensionality reduction
print("Applying t-SNE...")
perplexity = min(30, len(all_features_scaled)-1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_result = tsne.fit_transform(all_features_scaled)

# Apply K-means clustering
print("Performing K-means clustering...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_features_scaled)

# Create visualizations
plt.figure(figsize=(15, 12))

# 1. PCA visualization (colored by true labels)
plt.subplot(2, 2, 1)
for i, label in enumerate(['Negative', 'Positive']):
    plt.scatter(
        pca_result[labels == i, 0], 
        pca_result[labels == i, 1],
        label=label, 
        alpha=0.7
    )
plt.title('PCA Projection (Colored by True Labels)')
plt.xlabel(f'Principal Component 1 ({var_explained[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({var_explained[1]:.2%} variance)')
plt.legend()

# 2. PCA visualization (colored by cluster labels)
plt.subplot(2, 2, 2)
for i, label in enumerate(['Cluster 1', 'Cluster 2']):
    plt.scatter(
        pca_result[cluster_labels == i, 0], 
        pca_result[cluster_labels == i, 1],
        label=label, 
        alpha=0.7
    )
plt.title('PCA Projection (Colored by Clusters)')
plt.xlabel(f'Principal Component 1 ({var_explained[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({var_explained[1]:.2%} variance)')
plt.legend()

# 3. t-SNE visualization (colored by true labels)
plt.subplot(2, 2, 3)
for i, label in enumerate(['Negative', 'Positive']):
    plt.scatter(
        tsne_result[labels == i, 0], 
        tsne_result[labels == i, 1],
        label=label, 
        alpha=0.7
    )
plt.title('t-SNE Projection (Colored by True Labels)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

# 4. t-SNE visualization (colored by cluster labels)
plt.subplot(2, 2, 4)
for i, label in enumerate(['Cluster 1', 'Cluster 2']):
    plt.scatter(
        tsne_result[cluster_labels == i, 0], 
        tsne_result[cluster_labels == i, 1],
        label=label, 
        alpha=0.7
    )
plt.title('t-SNE Projection (Colored by Clusters)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.tight_layout()
plt.savefig('distance_embedding_2d_clustering.png', dpi=300)
plt.show()

# Calculate clustering evaluation metrics
ari = adjusted_rand_score(labels, cluster_labels)
nmi = normalized_mutual_info_score(labels, cluster_labels)

print(f"\nClustering Evaluation:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# Feature importance analysis
feature_names = [
    'Global Mean', 'Global Std Dev', 'Global Median', 'Global 25th Percentile', 'Global 75th Percentile',
    'CDR3a-Peptide Interface Mean', 'CDR3b-Peptide Interface Mean', 
    'CDR3a-Peptide Interface Std Dev', 'CDR3b-Peptide Interface Std Dev'
]

plt.figure(figsize=(10, 6))
loadings = np.abs(pca.components_[0])
sorted_idx = np.argsort(loadings)[::-1]
plt.barh(range(len(feature_names)), loadings[sorted_idx])
plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Contribution to Principal Component 1')
plt.xlabel('Feature Importance (Absolute Value)')
plt.tight_layout()
plt.savefig('distance_2d.png', dpi=300)
plt.show()