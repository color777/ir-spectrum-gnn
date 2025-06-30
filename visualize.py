import torch
import matplotlib.pyplot as plt
from evaluate import evaluate_single_batch
from model import GINModel  # 或你的模型
from data_preprocess import IRDataset
from torch_geometric.loader import DataLoader
import config as cfg

def plot_example(pred, true):
    plt.figure(figsize=(10, 4))
    plt.plot(pred, label="Predicted", alpha=0.7)
    plt.plot(true, label="True", alpha=0.7)
    plt.title("IR Spectrum Prediction")
    plt.xlabel("Wavenumber Index")
    plt.ylabel("Absorbance")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    dataset = IRDataset(cfg.dataset["path"])
    loader = DataLoader(dataset, batch_size=1)
    model = GINModel(
        num_node_features=dataset.num_node_features,
        hidden_dim=cfg.model["hidden_dim"],
        output_dim=len(dataset[0].y),
        num_layers=cfg.model["num_layers"]
    )
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    pred, true = evaluate_single_batch(model, next(iter(loader)))
    plot_example(pred, true)

if __name__ == "__main__":
    main()
