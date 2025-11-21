import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
import seaborn as sns
from tqdm import tqdm

# Set paths
unseen_csv_path = "/home/xycui/project/af3_binding/dataset/test_immrep23_unseen.csv"  # Adjust to actual path
model_path = "path/to/model/checkpoint.pth"  # Adjust to actual model checkpoint path
data_dir = "/home/fit/wangxw/WORK/af3_binding/dataset/test_immrep23_unseen"

# Function to load the model
def load_model(checkpoint_path):
    from af3_binding.model import Model
    model = Model(do_mlm=False)  # Adjust parameters as needed
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

# Function to extract pairformer outputs (single embeddings)
def extract_embeddings(model, data_loader):
    device = next(model.parameters()).device
    all_embeddings = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Move batch to the same device as model
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass through the model, but we need to capture intermediate outputs
            # We'll modify this to extract the pairformer's output
            # This requires modifying the model's forward method to return intermediate outputs
            
            # First, run the feature extractor
            feature_outputs = model.features(**{**batch, 'do_mlm': False, 'training': False})
            single_embedding, pair_embedding, point_embedding, single_mask, pair_mask = feature_outputs
            
            # Then run the pairformer to get 'single'
            single, pair = model.pairformer(single_embedding, pair_embedding, single_mask, pair_mask)
            
            # Get binding predictions
            x = torch.cat([torch.mean(single, dim=1), point_embedding], dim=-1)
            binding_pred = model.head(x).squeeze(-1)
            
            # Save the embeddings, labels and predictions
            all_embeddings.append(single.cpu())
            all_labels.append(batch['label'].cpu())
            all_preds.append(binding_pred.cpu())
    
    return torch.cat(all_embeddings), torch.cat(all_labels), torch.cat(all_preds)

# Custom dataset for loading specific samples
class TCRDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_dir, peptide="QIKVRVDMV"):
        self.data = []
        self.data_dir = data_dir
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Filter for the specific peptide
        peptide_samples = df[df['Peptide'] == peptide]
        print(f"Found {len(peptide_samples)} samples with peptide {peptide}")
        
        # Process each sample
        for idx, row in peptide_samples.iterrows():
            sample_path = f"{data_dir}/{idx}/{idx}.pt"  # Assuming idx corresponds to the folder name
            if os.path.exists(sample_path):
                self.data.append({
                    'path': sample_path,
                    'label': int(row['Label']),
                    'row_idx': idx
                })
        
        print(f"Successfully loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_info = self.data[idx]
        sample_data = torch.load(sample_info['path'])
        sample_data['label'] = torch.tensor(sample_info['label'])
        sample_data['row_idx'] = sample_info['row_idx']
        return sample_data

# Main execution
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path)
    model.to(device)
    
    # Create dataset and dataloader for specific peptide
    dataset = TCRDataset(unseen_csv_path, data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels, predictions = extract_embeddings(model, dataloader)
    
    # Process embeddings for dimensionality reduction
    # Reshape embeddings to [n_samples, n_features]
    n_samples = embeddings.shape[0]
    n_features = embeddings.shape[1] * embeddings.shape[2]  # L * 128
    flat_embeddings = embeddings.reshape(n_samples, n_features).numpy()
    
    # Apply dimensionality reduction
    print("Applying dimensionality reduction...")
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flat_embeddings)
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(flat_embeddings)-1), random_state=42)
    tsne_result = tsne.fit_transform(flat_embeddings)
    
    # 3. UMAP
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(flat_embeddings)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Prepare color mapping
    labels_np = labels.numpy()
    colors = ['#FF5555' if label == 1 else '#5555FF' for label in labels_np]
    
    # Plot PCA
    axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7)
    axes[0].set_title('PCA Projection')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    
    # Plot t-SNE
    axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.7)
    axes[1].set_title('t-SNE Projection')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    
    # Plot UMAP
    axes[2].scatter(umap_result[:, 0], umap_result[:, 1], c=colors, alpha=0.7)
    axes[2].set_title('UMAP Projection')
    axes[2].set_xlabel('Dimension 1')
    axes[2].set_ylabel('Dimension 2')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF5555', label='Binding (1)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#5555FF', label='Non-binding (0)', markersize=10)
    ]
    for ax in axes:
        ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('peptide_clustering_analysis.png', dpi=300)
    plt.show()
    
    # Additional analysis: mean embedding values for each class
    binding_embeddings = embeddings[labels == 1]
    nonbinding_embeddings = embeddings[labels == 0]
    
    if len(binding_embeddings) > 0 and len(nonbinding_embeddings) > 0:
        mean_binding = binding_embeddings.mean(dim=0)
        mean_nonbinding = nonbinding_embeddings.mean(dim=0)
        
        # Plot heatmaps of mean embeddings
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(mean_binding.numpy(), cmap='viridis')
        plt.title('Mean Embedding for Binding Samples')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(mean_nonbinding.numpy(), cmap='viridis')
        plt.title('Mean Embedding for Non-binding Samples')
        
        plt.tight_layout()
        plt.savefig('mean_embeddings_comparison.png', dpi=300)
        plt.show()
        
        # Plot the difference
        plt.figure(figsize=(8, 6))
        embedding_diff = mean_binding - mean_nonbinding
        sns.heatmap(embedding_diff.numpy(), cmap='coolwarm', center=0)
        plt.title('Embedding Difference (Binding - Non-binding)')
        plt.tight_layout()
        plt.savefig('embedding_difference.png', dpi=300)
        plt.show()
    
if __name__ == "__main__":
    main()