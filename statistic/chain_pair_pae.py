import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
import seaborn as sns
from tqdm import tqdm

# Define data paths
pos_dir = "/home/xycui/project/af3_binding/dataset/pos_tcrpmhc"
neg_dir = "/home/xycui/project/af3_binding/dataset/neg_tcrpmhc"

def extract_interface_metrics(chain_pair_pae):
    """Extract interface-specific metrics from chain_pair_pae matrix"""
    # Order: TCRα(0), TCRβ(1), peptide(2), MHC(3)
    tcr_peptide = (chain_pair_pae[0, 2] + chain_pair_pae[1, 2]+chain_pair_pae[2, 0] + chain_pair_pae[2, 1]) / 4  # TCR-peptide
    tcr_mhc = (chain_pair_pae[0, 3] + chain_pair_pae[1, 3]+chain_pair_pae[3, 0] + chain_pair_pae[3, 1]) / 2      # TCR-MHC
    peptide_mhc = chain_pair_pae[2, 3]+chain_pair_pae[3, 2] /2      # peptide-MHC
    
    return {
        'TCR_peptide': tcr_peptide,
        'TCR_MHC': tcr_mhc,
        'peptide_MHC': peptide_mhc
    }

def analyze_summary_pae():
    """Analyze PAE differences from summary files"""
    # Find all summary confidence files
    pos_files = glob(os.path.join(pos_dir, "*/*_summary_confidences.json"))
    neg_files = glob(os.path.join(neg_dir, "*/*_summary_confidences.json"))
    
    print(f"Found {len(pos_files)} positive and {len(neg_files)} negative summary files")
    
    # Store interface metrics
    pos_metrics = {'TCR_peptide': [], 'TCR_MHC': [], 'peptide_MHC': []}
    neg_metrics = {'TCR_peptide': [], 'TCR_MHC': [], 'peptide_MHC': []}
    
    # Process positive samples
    for file in tqdm(pos_files, desc="Processing positive samples"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            chain_pair_pae = np.array(data['chain_pair_pae_min'])
            metrics = extract_interface_metrics(chain_pair_pae)
            
            for key, value in metrics.items():
                pos_metrics[key].append(value)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Process negative samples
    for file in tqdm(neg_files, desc="Processing negative samples"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            chain_pair_pae = np.array(data['chain_pair_pae_min'])
            metrics = extract_interface_metrics(chain_pair_pae)
            
            for key, value in metrics.items():
                neg_metrics[key].append(value)
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Statistical analysis and visualization
    interfaces = ['TCR_peptide', 'TCR_MHC', 'peptide_MHC']
    
    # Create multi-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams.update({'font.size': 12})
    
    for i, interface in enumerate(interfaces):
        # Get data
        pos_data = pos_metrics[interface]
        neg_data = neg_metrics[interface]
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(pos_data, neg_data, equal_var=False)
        pos_mean, neg_mean = np.mean(pos_data), np.mean(neg_data)
        pos_std, neg_std = np.std(pos_data), np.std(neg_data)
        
        print(f"\n{interface.replace('_', '-')} Interface:")
        print(f"  Positive: {pos_mean:.4f} ± {pos_std:.4f}")
        print(f"  Negative: {neg_mean:.4f} ± {neg_std:.4f}")
        print(f"  Difference: {pos_mean - neg_mean:.4f}")
        print(f"  p-value: {p_val:.6f} ({'Significant' if p_val < 0.05 else 'Not significant'})")
        
        # Plot density distribution
        ax = axes[i]
        sns.kdeplot(pos_data, color='red', label='Positive', fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(neg_data, color='blue', label='Negative', fill=True, alpha=0.3, ax=ax)
        
        # Add annotations
        ax.set_title(f'{interface.replace("_", "-")} Interface', fontsize=14)
        ax.set_xlabel('Minimum PAE (Å)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        
        # Add statistical annotation
        y_max = ax.get_ylim()[1]
        if p_val < 0.001:
            sig_text = "p < 0.001 ***"
        elif p_val < 0.01:
            sig_text = f"p = {p_val:.3f} **"
        elif p_val < 0.05:
            sig_text = f"p = {p_val:.3f} *"
        else:
            sig_text = f"p = {p_val:.3f} (n.s.)"
        
        ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, 
                ha='center', va='top', fontsize=12)
        
        ax.legend(fontsize=12)
    
    plt.suptitle('AF3 Predicted Minimum PAE Comparison at Different Interfaces', fontsize=16)
    plt.tight_layout()
    plt.savefig('chain_pairpae_comparison.png', dpi=300)
    print("\nAnalysis complete. Visualization saved to 'interface_pae_comparison.png'.")

if __name__ == "__main__":
    analyze_summary_pae()