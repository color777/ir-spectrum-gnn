import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import json
import joblib
import pickle
import torch

def smooth_curve(y, window_size=15, poly_order=3):
    if len(y) < window_size:
        return y
    return savgol_filter(y, window_size, poly_order)

def normalize_spectra(spectra_array):
    scaler = StandardScaler()
    return scaler.fit_transform(spectra_array)

def minmax_scale(spectra_array):
    min_val = np.min(spectra_array, axis=1, keepdims=True)
    max_val = np.max(spectra_array, axis=1, keepdims=True)
    return (spectra_array - min_val) / (max_val - min_val + 1e-8)

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

def save_pca_scaler(pca, scaler, pca_path="pca_model.pkl", scaler_path="scaler.pkl"):
    with open(pca_path, "wb") as f1, open(scaler_path, "wb") as f2:
        pickle.dump(pca, f1)
        pickle.dump(scaler, f2)

def load_pca_scaler(pca_path="pca_model.pkl", scaler_path="scaler.pkl"):
    with open(pca_path, "rb") as f1, open(scaler_path, "rb") as f2:
        pca = pickle.load(f1)
        scaler = pickle.load(f2)
    return pca, scaler

# ✅ 稳定版 SID 损失函数，避免 log(0) 和 nan
def spectral_information_divergence(y_pred, y_true, eps=1e-8):
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(0)

    # 强制非负 + 避免除以0
    y_pred = y_pred.clamp(min=eps)
    y_true = y_true.clamp(min=eps)

    y_pred = y_pred / (y_pred.sum(dim=1, keepdim=True) + eps)
    y_true = y_true / (y_true.sum(dim=1, keepdim=True) + eps)

    sid = (y_pred * torch.log(y_pred / y_true) +
           y_true * torch.log(y_true / y_pred))
    return sid.sum(dim=1).mean()
