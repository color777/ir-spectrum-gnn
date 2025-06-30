import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
from utils import smooth_curve

class IRDataset(InMemoryDataset):
    def __init__(self, csv_file, target_len=6800, normalize=True, crop_range=(500, 3800), use_smooth=False, clip_max=None):
        super().__init__()
        self.data_list = []
        self.scaler = StandardScaler() if normalize else None

        df = pd.read_csv(csv_file)
        raw_spectra = []

        print("📦 正在加载数据并构建图...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                continue
            Chem.Kekulize(mol, clearAromaticFlags=True)

            edge_index = [[], []]
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index[0] += [a, b]
                edge_index[1] += [b, a]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float32)

            spec = np.array(row.values[1:], dtype=np.float32)
            if len(spec) > target_len:
                spec = spec[:target_len]
            elif len(spec) < target_len:
                spec = np.pad(spec, (0, target_len - len(spec)), mode='constant')

            if crop_range:
                start, end = crop_range
                if len(spec) < end:
                    continue
                spec = spec[start:end]

            if clip_max:
                spec = np.clip(spec, 0, clip_max)

            if use_smooth:
                spec = smooth_curve(spec)

            fp = spec[0:1000]
            sum_fp = fp.sum()
            if sum_fp > 0:
                spec = spec / sum_fp

            raw_spectra.append(spec)
            self.data_list.append(Data(x=x, edge_index=edge_index))

        assert all(len(s) == len(raw_spectra[0]) for s in raw_spectra), "❌ 存在光谱长度不一致！"

        raw_spectra = np.array(raw_spectra)
        if normalize:
            raw_spectra = self.scaler.fit_transform(raw_spectra)
            joblib.dump(self.scaler, "scaler.pkl")

        for i, data in enumerate(self.data_list):
            data.y = torch.tensor(raw_spectra[i], dtype=torch.float32)

        self.data, self.slices = self.collate(self.data_list)
        print(f"✅ 最终图样本数: {len(self.data_list)}")

    @property
    def num_node_features(self):
        return 1
