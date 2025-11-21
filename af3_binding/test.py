import os
import os.path as osp
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from Bio.PDB import MMCIFParser
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve

from visualizer import get_local
get_local.activate() # 激活装饰器

# 添加项目根目录到系统路径
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import Model
from model.features import Features
from data.iedb_data import collate_fn

get_index = []

class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = yaml.safe_load(f)

    def encode(self, text, as_one_token=True):
        if as_one_token:
            return [self.vocab.get(text, self.vocab['unknown'])]
        return [self.vocab.get(token, self.vocab['<unk>']) for token in text]


class TestDataset(Dataset):
    def __init__(self, csv_file, test_dir,
                 va_vocab, ja_vocab, vb_vocab, jb_vocab,
                 hla_vocab, peptide_vocab):
        self.data_df_full = pd.read_csv(csv_file)
        # 过滤掉没有对应文件夹的样本
        valid_data = []
        for i in range(len(self.data_df_full)):
            if os.path.exists(os.path.join(test_dir, str(i))):
                row = self.data_df_full.iloc[i].copy()
                row['original_idx'] = i
                valid_data.append(row)

        # 创建新的DataFrame
        self.data_df = pd.DataFrame(valid_data)
        print(f"有效样本数: {len(self.data_df)}/{len(self.data_df_full)}")
        # print(self.data_df)
        self.test_dir = test_dir
        self.va_tokenizer = Tokenizer(va_vocab)
        self.ja_tokenizer = Tokenizer(ja_vocab)
        self.vb_tokenizer = Tokenizer(vb_vocab)
        self.jb_tokenizer = Tokenizer(jb_vocab)
        self.hla_tokenizer = Tokenizer(hla_vocab)
        self.peptide_tokenizer = Tokenizer(peptide_vocab)
        self.parser = MMCIFParser()

    def __len__(self):
        return len(self.data_df)

    def _distance_embedding(self, distance):
        centers = torch.linspace(2., 22., 16)  # [16]
        width = (22. - 2.) / 16
        rbf = torch.exp(-0.5 * ((distance[..., None] - centers) / width) ** 2)  # [L, L, 16, 16]
        return rbf.reshape(rbf.size(0), rbf.size(1), -1)  # [L, L, 256]

    def _get_distance_embedding(self, model):
        coords = []
        plddts = []
        for chain in model:
            for residue in chain:
                coords.append([residue[atom].get_coord() for atom in ['N', 'CA', 'C', 'O']])
                plddts.append([residue[atom].bfactor for atom in ['N', 'CA', 'C', 'O']])
        coords = np.array(coords)
        coords = torch.tensor(coords, dtype=torch.float32)  # [L, 4, 3]
        plddts = torch.tensor(plddts, dtype=torch.float32)  # [L, 4]
        plddts = torch.mean(plddts, dim=-1)  # [L]
        # distance matrix: [L, L, 4, 4]
        distance = torch.norm(coords[None, :, None, :, :] - coords[:, None, :, None, :], dim=-1)  # [1, L, 1, 4, 3] - [L, 1, 4, 1, 3] -> [L, L, 4, 4, 3] -> [L, L, 4, 4]
        # [L, L, 4, 4] -> [L, L, 16]
        distance = distance.reshape(distance.size(0), distance.size(1), -1)
        return self._distance_embedding(distance), plddts  # [L, L, 256], [L]

    def _get_cache_path(self, idx):
        original_idx = self.data_df.iloc[idx]['original_idx']
        sample_dir = osp.join(self.test_dir, str(original_idx))
        cache_file = osp.join(sample_dir, f"{original_idx}_token.pt")
        return cache_file

    def __getitem__(self, idx):
        cache_path = self._get_cache_path(idx)

        global get_index
        # 如果缓存文件已存在，直接加载
        if osp.exists(cache_path):
            # print(f"Loading cached data from {cache_path}")
            return torch.load(cache_path)
 
        # 如果缓存文件不存在，正常加载和处理数据
        row = self.data_df.iloc[idx]
        original_idx = row['original_idx']
        
        sample_dir = osp.join(self.test_dir, str(original_idx))
        if not osp.exists(sample_dir):
            
            raise FileNotFoundError(f"Test sample directory not found: {sample_dir}")
            

        embedding_path = osp.join(sample_dir, f"seed-1234_embeddings/{original_idx}_seed-1234_embeddings.npz")
        confidence_path = osp.join(sample_dir, f"{original_idx}_confidences.json")
        model_path = osp.join(sample_dir, f"{original_idx}_model.cif")
        
        for path in [embedding_path, confidence_path, model_path]:
            if not osp.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # 获取数据
        cdr3a = row['CDR3a'] if not pd.isna(row['CDR3a']) else ""
        cdr3b = row['CDR3b'] if not pd.isna(row['CDR3b']) else ""
        peptide = row['Peptide'] if not pd.isna(row['Peptide']) else ""
        
        tcra = row['TCRa'] if not pd.isna(row['TCRa']) else ""
        tcrb = row['TCRb'] if not pd.isna(row['TCRb']) else ""
        
        va = row['Va'] if not pd.isna(row['Va']) else "unknown" 
        ja = row['Ja'] if not pd.isna(row['Ja']) else "unknown"
        vb = row['Vb'] if not pd.isna(row['Vb']) else "unknown"
        jb = row['Jb'] if not pd.isna(row['Jb']) else "unknown"
        hla = row['HLA'] if not pd.isna(row['HLA']) else "unknown"
        
        embedding = np.load(embedding_path)
        with open(confidence_path, 'r') as f:
            confidence = json.load(f)
        model = self.parser.get_structure('model', model_path)[0]

        # sequence encoding
        seq = cdr3a + cdr3b + peptide
        seq_tokens = self.peptide_tokenizer.encode(seq, as_one_token=False)  # [L]
        seq_tokens = torch.tensor(seq_tokens, dtype=torch.long)

        s0 = tcra.find(cdr3a) if cdr3a in tcra else 0
        s1 = len(tcra) + (tcrb.find(cdr3b) if cdr3b in tcrb else 0)
        s2 = len(tcra) + len(tcrb)

        embedding_single = torch.tensor(embedding['single_embeddings'], dtype=torch.float32)  # [L_, 384]
        embedding_pair = torch.tensor(embedding['pair_embeddings'], dtype=torch.float32)  # [L_, L_, 128]

        # 提取 cdr3a, cdr3b, peptide 的 embedding
        consider = list(range(s0, s0 + len(cdr3a))) + list(range(s1, s1 + len(cdr3b))) + list(range(s2, s2 + len(peptide)))
        embedding_single = embedding_single[consider]  # [L, 384]
        embedding_pair = embedding_pair[consider][:, consider]  # [L, L, 128]

        chain_encoding = [1 for _ in range(len(cdr3a))] + [2 for _ in range(len(cdr3b))] + [3 for _ in range(len(peptide))]
        chain_encoding = torch.tensor(chain_encoding, dtype=torch.long)  # [L]

        # gene/allele encoding
        va_token = torch.tensor(self.va_tokenizer.encode(va), dtype=torch.long)  # [1]
        ja_token = torch.tensor(self.ja_tokenizer.encode(ja), dtype=torch.long)  # [1]
        vb_token = torch.tensor(self.vb_tokenizer.encode(vb), dtype=torch.long)  # [1]
        jb_token = torch.tensor(self.jb_tokenizer.encode(jb), dtype=torch.long)  # [1]
        hla_token = torch.tensor(self.hla_tokenizer.encode(hla), dtype=torch.long)  # [1]

        # structure encoding
        distance_embedding, plddts = self._get_distance_embedding(model)
        plddts = plddts[consider]
        distance_embedding = distance_embedding[consider][:, consider]  # [L, L, 256]
        pae = torch.tensor(confidence['pae'], dtype=torch.float32)
        pae_embedding = self._distance_embedding(pae)
        pae_embedding = pae_embedding[consider][:, consider]  # [L, L, 16]

        # 构建返回值
        sample = {
            # [tcra + tcrb + peptide]
            'seq_tokens': seq_tokens,  # [L]
            'embedding_single': embedding_single / 1000.,  # [L, 384]
            'embedding_pair': embedding_pair / 1000.,  # [L, L, 128]
            'chain_encoding': chain_encoding,  # [L]
            'distance_embedding': distance_embedding,  # [L, L, 256]
            'pae_embedding': pae_embedding,  # [L, L, 16]
            'plddts': plddts,  # [L]

            'va_token': va_token,
            'ja_token': ja_token,
            'vb_token': vb_token,
            'jb_token': jb_token,
            'hla_token': hla_token,

            'label': torch.tensor(int(row['Label']))
        }

        # 保存到缓存文件
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        torch.save(sample, cache_path)

        return sample


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    batch_size = 0
    count = 0
    target = [1,2,3,4,5,6,7,8,60,659]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估进度", ncols=100):
            # 将数据移到 GPU
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            
            # 前向传播

            if count in target:
                model.moe_mhc.moe_hook = None

            outputs = model(batch_device)

            count += 1
            
            scores =outputs['binding_pred']

            if count in target:
                model.moe_mhc.plot_moe(str(count))
                print(count,torch.sigmoid(scores))

            preds = torch.sigmoid(scores) > 0.5
            
            # 收集结果
            all_scores.extend(torch.sigmoid(scores).cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    
    # 对正样本和负样本的数量进行统计
    pos_count = sum(all_labels)
    neg_count = len(all_labels) - pos_count
    
    # 打印结果
    print(f"正样本数量: {pos_count}, 负样本数量: {neg_count}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"AUPR: {ap:.4f}")
    
    # 计算ROC和PR曲线
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
    
    # 绘制ROC曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {auc:.4f})')
    
    # 绘制PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
    
    plt.tight_layout()
    plt.savefig('./results/AUC.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ap': ap,
        'predictions': all_preds,
        'scores': all_scores,
        'labels': all_labels
    }

def load_config_from_checkpoint_dir(checkpoint_dir):
    """
    从检查点目录加载配置文件
    
    Args:
        checkpoint_dir: 检查点目录路径，包含config.yml文件
    
    Returns:
        配置字典
    """
    config_path = os.path.join(checkpoint_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

# 使用示例
# checkpoint_dir = "/home/xycui/project/af3_binding/results/tcr_pmhc_emb"
# config = load_config_from_checkpoint_dir(checkpoint_dir)

def main():
    # 设置设备
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #structure
    # model_name = "4block_3lossweight"
    # epoch_name = 'epoch_20'
    # seq
    # model_name = "seq_only"
    # epoch_name = 'epoch_30'
    
    # model_name = "seq_plus_structure_6.23"
    model_name = "seq_plus_structure_9.23"     # 模型名称
    epoch_name = 'epoch_10'                
    # model_name = "seq_pus_structure_6.10_WBCE"
    # 配置路径
    test_csv = "./dataset/test_immrep23_unseen.csv"
    test_dir = "./dataset/test_immrep23_unseen"
    model_weights = "./results/"+ model_name +"/model_weights/"+epoch_name+".pt"
    checkpoint_dir = "./results/"+ model_name 
    config = load_config_from_checkpoint_dir(checkpoint_dir)
    feature_config = config["model"]["feature_config"]
    model_config = config["model"]["model_config"]
    hla_config = config["model"]["hla_config"]
    # mlm_config = config["model"]["mlm_config"]
    mlm_config = None

    
    # 词汇表路径
    vocab_dir = "/home/xycui/project/af3_binding/config"
    va_vocab = os.path.join(vocab_dir, "va_config.yml")
    ja_vocab = os.path.join(vocab_dir, "ja_config.yml")
    vb_vocab = os.path.join(vocab_dir, "vb_config.yml")
    jb_vocab = os.path.join(vocab_dir, "jb_config.yml")
    hla_vocab = os.path.join(vocab_dir, "hla_config.yml")
    peptide_vocab = os.path.join(vocab_dir, "alphabet.yml")
    
    # 创建数据集
    test_dataset = TestDataset(
        test_csv, test_dir,
        va_vocab, ja_vocab, vb_vocab, jb_vocab,
        hla_vocab, peptide_vocab
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    
    i = 0
    # for batch in test_loader:
    #     i += 1
    #     if i == 16:
    #         for name in batch.keys():
    #             print(name,batch[name][4])
            # print(batch['peptide_tokens'][43],batch['label'][43],batch['cdr3b_tokens'][43])
        # 创建模型
    model = Model(feature_config, model_config,hla_config,mlm_config)
    

    # 加载预训练权重
    checkpoint = torch.load(model_weights, map_location=device)
    print(f"检查点中的键: {list(checkpoint.keys())}")
    # 不使用'model_state_dict'键，而是直接加载整个状态字典
    try:
        model.load_state_dict(checkpoint)
        print(f"成功加载模型权重：直接加载状态字典")
    except Exception as e:
        print(f"尝试直接加载状态字典失败: {e}")
        # 如果加载失败，尝试非严格模式
        model.load_state_dict(checkpoint, strict=False)
        print(f"成功加载模型权重：使用非严格模式")
    # model.load_state_dict(checkpoint['model_state_dict'])

    # model.moe_mhc.moe_hook = True

    model = model.to(device)
    print(f"Loaded model weights from {model_weights}")
    
    # 评估模型
    results = evaluate(model, test_loader, device)
    
    # model.moe_mhc.plot_moe(model_name)

    # 保存预测结果
    df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted': results['predictions'],
        'score': results['scores']
    })
    df.to_csv('./results/' + model_name + '/'+epoch_name+'predict.csv', index=False)
    print("Saved predictions to test_predictions.csv")

if __name__ == "__main__":
    main()