import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from data_preprocess import IRDataset
from model import GATv2_JK_Model, GIN_JK_Model, GCN_JK_Model
from train import train
from evaluate import evaluate_ensemble
import config as cfg

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # === 加载数据 ===
    df = pd.read_csv(cfg.dataset["path"])
    target_len = len(df.columns) - 1

    dataset = IRDataset(
        csv_file=cfg.dataset["path"],
        target_len=target_len,
        normalize=True,
        pca_dim=cfg.model["output_dim"],
        crop_range=(500, 3800),
        use_smooth=cfg.dataset.get("use_smooth", False),
        clip_max=cfg.dataset.get("clip_max", None)
    )
    print(f"✅ Loaded {len(dataset)} samples. PCA dim: {cfg.model['output_dim']}")

    # ✅ 保存 PCA 和 Scaler
    dataset.save_pca_and_scaler()

    # === 划分数据集 ===
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.train["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train["batch_size"])

    # === 初始化多个模型 ===
    models = []
    model_classes = [GATv2_JK_Model, GIN_JK_Model, GCN_JK_Model]

    for i, model_class in enumerate(model_classes):
        print(f"\n🧠 Training model {i+1}/{len(model_classes)}: {model_class.__name__}")
        model = model_class(
            num_node_features=dataset.num_node_features,
            hidden_dim=cfg.model["hidden_dim"],
            output_dim=cfg.model["output_dim"],
            num_layers=cfg.model["num_layers"]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train["lr"], weight_decay=cfg.train["weight_decay"])
        result = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            epochs=cfg.train["epochs"],
            device=device,
            weight_decay=cfg.train["weight_decay"]
        )
        torch.save(model.state_dict(), f"best_model_{i+1}.pth")
        models.append(model)

        if result:
            train_losses, val_losses = result
            plt.figure(figsize=(7, 4))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.title(f"Training Curve: {model_class.__name__}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # === 加载模型参数（保证使用 best_model）===
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(f"best_model_{i+1}.pth"))
        model.to(device)

    # === 集成评估 ===
    evaluate_ensemble(models, val_loader, device, use_pca=True)

if __name__ == "__main__":
    main()
