import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from tqdm import tqdm
import seaborn as sns

# Define paths
neg_dir = '/home/xycui/project/af3_binding/dataset/neg_tcrpmhc_pt'
pos_dir = '/home/xycui/project/af3_binding/dataset/pos_tcrpmhc_pt'

def collect_distance_embeddings(root_dir, limit=5000):
    """Collect distance embeddings from .pt files"""
    all_embeddings = []
    all_files = []
    
    # Find all .pt files in the directory (recursively)
    for pt_file in tqdm(glob.glob(f"{root_dir}/**/*.pt", recursive=True)[:limit]):
        all_files.append(pt_file)
        try:
            # Load the pytorch file
            data = torch.load(pt_file)
            # Extract the distance embedding
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

# Calculate statistics for visualization
def compute_embedding_stats(embeddings):
    """Compute statistics from distance embeddings"""
    # Calculate mean value for each sample
    mean_values = [emb.mean().item() for emb in embeddings]
    
    # Calculate mean of specific regions (e.g., between CDR3 and peptide)
    cdr_pep_means = []
    for emb in embeddings:
        L = emb.shape[0]
        # Assuming the last portion is peptide (based on your data loading code)
        peptide_start = 2*L//3  # Rough approximation, adjust if needed
        
        # Region between CDR3 and peptide
        cdr_pep_region = emb[:peptide_start, peptide_start:]
        cdr_pep_means.append(cdr_pep_region.mean().item())
    
    # Calculate variance of embeddings
    var_values = [emb.var().item() for emb in embeddings]
    
    return {
        'means': mean_values,
        'cdr_pep_means': cdr_pep_means,
        'variances': var_values
    }

# Compute statistics
neg_stats = compute_embedding_stats(neg_embeddings)
pos_stats = compute_embedding_stats(pos_embeddings)

# Visualization
plt.figure(figsize=(15, 10))

# 1. Overall mean distribution
plt.subplot(2, 2, 1)
sns.histplot(neg_stats['means'], color='red', alpha=0.5, label='Negative')
sns.histplot(pos_stats['means'], color='blue', alpha=0.5, label='Positive')
plt.title('Distribution of Mean Distance Embedding Values')
plt.xlabel('Mean Value')
plt.ylabel('Count')
plt.legend()

# 2. CDR-peptide interface means
plt.subplot(2, 2, 2)
sns.histplot(neg_stats['cdr_pep_means'], color='red', alpha=0.5, label='Negative')
sns.histplot(pos_stats['cdr_pep_means'], color='blue', alpha=0.5, label='Positive')
plt.title('Mean Distance at CDR3-Peptide Interface')
plt.xlabel('Mean Value')
plt.ylabel('Count')
plt.legend()

# 3. Variance distribution
plt.subplot(2, 2, 3)
sns.histplot(neg_stats['variances'], color='red', alpha=0.5, label='Negative')
sns.histplot(pos_stats['variances'], color='blue', alpha=0.5, label='Positive')
plt.title('Distribution of Distance Embedding Variance')
plt.xlabel('Variance')
plt.ylabel('Count')
plt.legend()

# 4. Boxplot comparison
plt.subplot(2, 2, 4)
data = {
    'Overall Mean (Neg)': neg_stats['means'],
    'Overall Mean (Pos)': pos_stats['means'],
    'Interface Mean (Neg)': neg_stats['cdr_pep_means'],
    'Interface Mean (Pos)': pos_stats['cdr_pep_means']
}
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title('Comparison of Distance Embedding Statistics')
plt.ylabel('Value')

plt.tight_layout()
plt.savefig('distance_embedding_comparison.png', dpi=300)
plt.show()

# Print statistical summary
print("\nStatistical Summary:")
print(f"Negative Samples (n={len(neg_embeddings)}):")
print(f"  Mean of means: {np.mean(neg_stats['means']):.4f} ± {np.std(neg_stats['means']):.4f}")
print(f"  Mean at interface: {np.mean(neg_stats['cdr_pep_means']):.4f} ± {np.std(neg_stats['cdr_pep_means']):.4f}")

print(f"\nPositive Samples (n={len(pos_embeddings)}):")
print(f"  Mean of means: {np.mean(pos_stats['means']):.4f} ± {np.std(pos_stats['means']):.4f}")
print(f"  Mean at interface: {np.mean(pos_stats['cdr_pep_means']):.4f} ± {np.std(pos_stats['cdr_pep_means']):.4f}")