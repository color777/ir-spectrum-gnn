import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import json
import joblib
import pickle

# ========== 平滑与标准化 ==========
def smooth_curve(y, window_size=15, poly_order=3):
    if len(y) < window_size:
        return y  # 太短不处理
    return savgol_filter(y, window_size, poly_order)

def normalize_spectra(spectra_array):
    scaler = StandardScaler()
    return scaler.fit_transform(spectra_array)

def minmax_scale(spectra_array):
    min_val = np.min(spectra_array, axis=1, keepdims=True)
    max_val = np.max(spectra_array, axis=1, keepdims=True)
    return (spectra_array - min_val) / (max_val - min_val + 1e-8)

# ========== 配置读取 ==========
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ========== 标签标准化 ==========
class LabelScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, y):
        self.scaler.fit(y)

    def transform(self, y):
        return self.scaler.transform(y)

    def inverse_transform(self, y):
        return self.scaler.inverse_transform(y)

    def save(self, path="label_scaler.pkl"):
        joblib.dump(self.scaler, path)

    def load(self, path="label_scaler.pkl"):
        self.scaler = joblib.load(path)

# ========== PCA 与 Scaler 保存/加载 ==========
def save_pca_scaler(pca, scaler, pca_path="pca_model.pkl", scaler_path="scaler.pkl"):
    """保存 PCA 和 StandardScaler 模型到本地文件"""
    with open(pca_path, "wb") as f1, open(scaler_path, "wb") as f2:
        pickle.dump(pca, f1)
        pickle.dump(scaler, f2)

def load_pca_scaler(pca_path="pca_model.pkl", scaler_path="scaler.pkl"):
    """加载已保存的 PCA 和 Scaler 模型"""
    with open(pca_path, "rb") as f1, open(scaler_path, "rb") as f2:
        pca = pickle.load(f1)
        scaler = pickle.load(f2)
    return pca, scaler
