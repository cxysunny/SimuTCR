import os
import os.path as osp

import json
import yaml
import numpy as np

import torch
import pandas as pd
from Bio.PDB import MMCIFParser
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def get_train_and_test_dataset(data_dir, test_size=0.2, seed=1234, **kwargs):
    iedb_pos = pd.read_csv(osp.join(data_dir, 'iedb_positives.csv'))
    iedb_neg = pd.read_csv(osp.join(data_dir, 'iedb_negatives.csv'))
    data = []
    for i in range(len(iedb_pos)):
        if not osp.exists(osp.join(data_dir, 'pos_tcrpmhc', f'{i}')):
            continue
        data.append(
            {
                'tcra': iedb_pos.loc[i, 'TCRa'],
                'tcrb': iedb_pos.loc[i, 'TCRb'],
                'cdr3a': iedb_pos.loc[i, 'CDR3a'],
                'cdr3b': iedb_pos.loc[i, 'CDR3b'],
                'hla': iedb_pos.loc[i, 'HLA'],
                'va': iedb_pos.loc[i, 'Va'],
                'ja': iedb_pos.loc[i, 'Ja'],
                'vb': iedb_pos.loc[i, 'Vb'],
                'jb': iedb_pos.loc[i, 'Jb'],
                'peptide': iedb_pos.loc[i, 'Peptide'],

                'embedding_dir': osp.join(data_dir, 'pos_tcrpmhc', f'{i}/seed-{seed}_embeddings/{i}_seed-{seed}_embeddings.npz'),
                'confidence_dir': osp.join(data_dir, 'pos_tcrpmhc', f'{i}/{i}_confidences.json'),
                'model_dir': osp.join(data_dir, 'pos_tcrpmhc', f'{i}/{i}_model.cif'),

                'label': 1
            }
        )
    for i in range(len(iedb_neg)):
        if not osp.exists(osp.join(data_dir, 'neg_tcrpmhc', f'{i}')):
            continue
        data.append(
            {
                'tcra': iedb_neg.loc[i, 'TCRa'],
                'tcrb': iedb_neg.loc[i, 'TCRb'],
                'cdr3a': iedb_neg.loc[i, 'CDR3a'],
                'cdr3b': iedb_neg.loc[i, 'CDR3b'],
                'hla': iedb_neg.loc[i, 'HLA'],
                'va': iedb_neg.loc[i, 'Va'],
                'ja': iedb_neg.loc[i, 'Ja'],
                'vb': iedb_neg.loc[i, 'Vb'],
                'jb': iedb_neg.loc[i, 'Jb'],
                'peptide': iedb_neg.loc[i, 'Peptide'],

                'embedding_dir': osp.join(data_dir, 'neg_tcrpmhc', f'{i}/seed-{seed}_embeddings/{i}_seed-{seed}_embeddings.npz'),
                'confidence_dir': osp.join(data_dir, 'neg_tcrpmhc', f'{i}/{i}_confidences.json'),
                'model_dir': osp.join(data_dir, 'neg_tcrpmhc', f'{i}/{i}_model.cif'),

                'label': 0
            }
        )
    
    # Shuffle data
    torch.manual_seed(seed)
    data = [data[i] for i in torch.randperm(len(data))]

    test_size = int(test_size * len(data))
    train_data = data[test_size:]
    test_data = data[:test_size]
    train_dataset = IEDBDataset(train_data, **kwargs)
    test_dataset = IEDBDataset(test_data, **kwargs)
    return train_dataset, test_dataset

class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = yaml.safe_load(f)

    def encode(self, text, as_one_token=True):
        if as_one_token:
            return [self.vocab.get(text, self.vocab['unknown'])]
        return [self.vocab.get(token, self.vocab['<unk>']) for token in text]


class IEDBDataset(Dataset):
    def __init__(self, data,
                 va_vocab: str, ja_vocab: str,
                 vb_vocab: str, jb_vocab: str,
                 hla_vocab: str, peptide_vocab: str, **kwargs):
        self.data = data
        self.va_tokenizer = Tokenizer(va_vocab)
        self.ja_tokenizer = Tokenizer(ja_vocab)
        self.vb_tokenizer = Tokenizer(vb_vocab)
        self.jb_tokenizer = Tokenizer(jb_vocab)
        self.hla_tokenizer = Tokenizer(hla_vocab)
        self.peptide_tokenizer = Tokenizer(peptide_vocab)
        self.parser = MMCIFParser()

    def __len__(self):
        return len(self.data)

    def _distance_embedding(self, distance):
            centers = torch.linspace(2., 22., 16) # [16]
            width = (22. - 2.) / 16
            rbf = torch.exp(-0.5 * ((distance[..., None] - centers) / width) ** 2) # [L, L, 16, 16]
            return rbf.reshape(rbf.size(0), rbf.size(1), -1) # [L, L, 256]

    def _get_distance_embedding(self, model):
        coords = []
        plddts = []
        for chain in model:
            for residue in chain:
                coords.append([residue[atom].get_coord() for atom in ['N', 'CA', 'C', 'O']])
                plddts.append([residue[atom].bfactor for atom in ['N', 'CA', 'C', 'O']])
        coords = np.array(coords)
        coords = torch.tensor(coords, dtype=torch.float32) # [L, 4, 3]
        plddts = torch.tensor(plddts, dtype=torch.float32) # [L, 4]
        plddts = torch.mean(plddts, dim=-1) # [L]
        # distance matrix: [L, L, 4, 4]
        distance = torch.norm(coords[None, :, None, :, :] - coords[:, None, :, None, :], dim=-1) # [1, L, 1, 4, 3] - [L, 1, 4, 1, 3] -> [L, L, 4, 4, 3] -> [L, L, 4, 4]
        # [L, L, 4, 4] -> [L, L, 16]
        distance = distance.reshape(distance.size(0), distance.size(1), -1)
        return self._distance_embedding(distance), plddts # [L, L, 256]

    def _get_cache_path(self, idx):
        
        meta_data = self.data[idx]
        # 当前 embedding_dir 是：
        # .../neg_tcrpmhc/0/seed-1234_embeddings/0_seed-1234_embeddings.npz
        # 我们需要获取上两级目录，即 .../neg_tcrpmhc/0/
        sample_dir = osp.dirname(osp.dirname(meta_data['embedding_dir']))
        # 然后从该路径获取样本 ID
        sample_id = osp.basename(sample_dir)
        # 缓存路径形如 .../neg_tcrpmhc/0/0.pt
        cache_file = osp.join(sample_dir, f"{sample_id}_token.pt")
        return cache_file

    def __getitem__(self, idx):
        cache_path = self._get_cache_path(idx)

        # 如果缓存文件已存在，直接加载
        if osp.exists(cache_path):
            # print(f"Loading cached data from {cache_path}")
            return torch.load(cache_path)

        # 如果缓存文件不存在，正常加载和处理数据
        meta_data = self.data[idx]

        tcra, tcrb = meta_data['tcra'], meta_data['tcrb']
        cdr3a, cdr3b, peptide = meta_data['cdr3a'], meta_data['cdr3b'], meta_data['peptide']
        va, ja, vb, jb = meta_data['va'], meta_data['ja'], meta_data['vb'], meta_data['jb']
        hla = meta_data['hla']

        embedding = np.load(meta_data['embedding_dir'])
        with open(meta_data['confidence_dir'], 'r') as f:
            confidence = json.load(f)
        model = self.parser.get_structure('model', meta_data['model_dir'])[0]
        # print(cdr3a, cdr3b, peptide)
        # sequence encoding
        seq = cdr3a + cdr3b + peptide

        seq_tokens = self.peptide_tokenizer.encode(seq, as_one_token=False)  # [L]
        seq_tokens = torch.tensor(seq_tokens, dtype=torch.long)
        cdr3a_tokens = self.peptide_tokenizer.encode(cdr3a, as_one_token=False)  # [L]
        cdr3a_tokens = torch.tensor(cdr3a_tokens, dtype=torch.long)  # [L]
        cdr3b_tokens = self.peptide_tokenizer.encode(cdr3b, as_one_token=False)  # [L]
        cdr3b_tokens = torch.tensor(cdr3b_tokens, dtype=torch.long)  # [L]
        peptide_tokens = self.peptide_tokenizer.encode(peptide, as_one_token=False)
        peptide_tokens = torch.tensor(peptide_tokens, dtype=torch.long)

        
        s0 = tcra.find(cdr3a)
        s1 = len(tcra) + tcrb.find(cdr3b)
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
        vb_token = torch.tensor(self.vb_tokenizer.encode(vb), dtype=torch.long)
        jb_token = torch.tensor(self.jb_tokenizer.encode(jb), dtype=torch.long)
        hla_token = torch.tensor(self.hla_tokenizer.encode(hla), dtype=torch.long)

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

            'cdr3a_tokens': cdr3a_tokens,
            'cdr3b_tokens': cdr3b_tokens,
            'peptide_tokens': peptide_tokens,

            'label': torch.tensor(meta_data['label'])
        }

        # 保存到缓存文件
        torch.save(sample, cache_path)

        return sample

def collate_fn(batch):
    batch.sort(key=lambda x: x['seq_tokens'].size(0), reverse=True)
    # single features, padding to [max_len, dim]
    seq_tokens = pad_sequence([x['seq_tokens'] for x in batch], batch_first=True, padding_value=0) #[B,L]
    cdr3a_tokens = pad_sequence([x['cdr3a_tokens'] for x in batch], batch_first=True, padding_value=0) #[B,L1]
    cdr3b_tokens = pad_sequence([x['cdr3b_tokens'] for x in batch], batch_first=True, padding_value=0) #[B,L2]
    peptide_tokens = pad_sequence([x['peptide_tokens'] for x in batch], batch_first=True, padding_value=0) #[B,L3]
    embedding_single = pad_sequence([x['embedding_single'] for x in batch], batch_first=True, padding_value=0) #[B,L,384]
    chain_encoding = pad_sequence([x['chain_encoding'] for x in batch], batch_first=True, padding_value=0) #[B,L]
    plddts = pad_sequence([x['plddts'] for x in batch], batch_first=True, padding_value=0) #[B,L]
    # pair features, padding to [max_len, max_len, dim]
    batch_size = len(batch)
    max_len = seq_tokens.size(1)
    embedding_pair = torch.zeros(batch_size, max_len, max_len, batch[0]['embedding_pair'].size(-1)) # [B,L,L,128]
    distance_embedding = torch.zeros(batch_size, max_len, max_len, batch[0]['distance_embedding'].size(-1)) # [B,L,L,256]
    pae_embedding = torch.zeros(batch_size, max_len, max_len, batch[0]['pae_embedding'].size(-1)) # [B,L,L,16]
    for i, x in enumerate(batch):
        embedding_pair[i, :x['embedding_pair'].size(0), :x['embedding_pair'].size(1)] = x['embedding_pair']
        distance_embedding[i, :x['distance_embedding'].size(0), :x['distance_embedding'].size(1)] = x['distance_embedding']
        pae_embedding[i, :x['pae_embedding'].size(0), :x['pae_embedding'].size(1)] = x['pae_embedding']
    # gene/allele features
    va_token = torch.cat([x['va_token'] for x in batch], dim=0)
    ja_token = torch.cat([x['ja_token'] for x in batch], dim=0)
    vb_token = torch.cat([x['vb_token'] for x in batch], dim=0)
    jb_token = torch.cat([x['jb_token'] for x in batch], dim=0)
    hla_token = torch.cat([x['hla_token'] for x in batch], dim=0) # [batch_size]
    # label
    label = torch.tensor([x['label'] for x in batch], dtype=torch.long) # [batch_size]
    return {
        'seq_tokens': seq_tokens, # [B,L]
        'embedding_single': embedding_single, # [B,L,384]
        'embedding_pair': embedding_pair, # [B,L,L,128]
        'chain_encoding': chain_encoding, # [B,L]
        'distance_embedding': distance_embedding, # [B,L,L,256]
        'pae_embedding': pae_embedding, # [B,L,L,16]
        'plddts': plddts, # [B,L]
        'va_token': va_token, # [batch_size]
        'ja_token': ja_token, # [batch_size]
        'vb_token': vb_token, # [batch_size]
        'jb_token': jb_token, # [batch_size]
        'hla_token': hla_token, # [batch_size]
        'cdr3a_tokens': cdr3a_tokens,
        'cdr3b_tokens': cdr3b_tokens,
        'peptide_tokens': peptide_tokens,
        'label': label # [batch_size]
    }


class IEDBDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

