import torch
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate(model, loader, device, use_pca=True):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.view(batch.num_graphs, -1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    if use_pca:
        try:
            from utils import load_pca_scaler
            pca, scaler = load_pca_scaler()
            preds = scaler.inverse_transform(pca.inverse_transform(preds))
            targets = scaler.inverse_transform(pca.inverse_transform(targets))
        except Exception as e:
            print("⚠️ PCA 反变换失败，使用原始输出。", e)

    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    print("📊 Evaluation Results:")
    print(f" - MSE: {mse:.6f}")
    print(f" - R² Score: {r2:.6f}")

def evaluate_ensemble(models, loader, device, use_pca=True):
    for model in models:
        model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outs = [model(batch).cpu().numpy() for model in models]
            out_avg = np.mean(outs, axis=0)
            preds.append(out_avg)
            targets.append(batch.y.view(batch.num_graphs, -1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    if use_pca:
        try:
            from utils import load_pca_scaler
            pca, scaler = load_pca_scaler()
            preds = scaler.inverse_transform(pca.inverse_transform(preds))
            targets = scaler.inverse_transform(pca.inverse_transform(targets))
        except Exception as e:
            print("⚠️ PCA 反变换失败，使用原始输出。", e)

    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    print("📊 [Ensemble] Evaluation Results:")
    print(f" - MSE: {mse:.6f}")
    print(f" - R² Score: {r2:.6f}")
