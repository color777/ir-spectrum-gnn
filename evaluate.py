import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            y = batch.y
            if y.ndim == 1:
                y = y.view(output.shape)

            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("📊 Evaluation Results:")
    print(f" - MSE: {mse:.6f}")
    print(f" - R² Score: {r2:.6f}")
    return mse, r2
