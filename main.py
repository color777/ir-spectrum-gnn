import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from data_preprocess import IRDataset
from model import GATv2_JK_Model
from train import train
from evaluate import evaluate
from utils import spectral_information_divergence
import config as cfg

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    df = pd.read_csv(cfg.dataset["path"])
    target_len = len(df.columns) - 1

    dataset = IRDataset(
        csv_file=cfg.dataset["path"],
        target_len=target_len,
        normalize=True,
        crop_range=(500, 3800),
        use_smooth=cfg.dataset.get("use_smooth", False),
        clip_max=cfg.dataset.get("clip_max", None)
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.train["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train["batch_size"])

    ensemble_r2s = []
    for i in range(cfg.train.get("ensemble", 1)):
        print(f"🚀 Training model {i+1}")
        model = GATv2_JK_Model(
            num_node_features=dataset.num_node_features,
            hidden_dim=cfg.model["hidden_dim"],
            output_dim=len(dataset[0].y),
            num_layers=cfg.model["num_layers"]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train["lr"], weight_decay=cfg.train["weight_decay"])
        result = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=spectral_information_divergence,
            epochs=cfg.train["epochs"],
            device=device,
            weight_decay=cfg.train["weight_decay"]
        )
        torch.save(model.state_dict(), f"best_model_{i}.pth")

        _, r2 = evaluate(model, val_loader, device)
        ensemble_r2s.append(r2)

    print(f"\n✅ Average R² over {len(ensemble_r2s)} models: {sum(ensemble_r2s)/len(ensemble_r2s):.4f}")

if __name__ == "__main__":
    main()
