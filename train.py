import torch
from tqdm import tqdm
import torch.nn.functional as F

class EarlyStopper:
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def check(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def hybrid_loss_fn(pred, target, alpha=0.8):
    mse_loss = F.mse_loss(pred, target)
    cosine_loss = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return alpha * mse_loss + (1 - alpha) * cosine_loss

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, weight_decay=1e-5):
    best_val_loss = float("inf")
    stopper = EarlyStopper(patience=20)
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            target = batch.y.view(batch.num_graphs, -1).to(device)

            loss = hybrid_loss_fn(output, target)

            # L2 正则
            l2_reg = sum(torch.norm(p) for p in model.parameters())
            loss += weight_decay * l2_reg

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"📈 Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                target = batch.y.view(batch.num_graphs, -1).to(device)
                val_loss += hybrid_loss_fn(output, target).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"🔍 Epoch {epoch}: Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ 模型保存成功！当前最优 Val Loss: {best_val_loss:.6f}")
        else:
            print(f"⚠️ 验证集损失未提升（当前: {avg_val_loss:.6f}, 最佳: {best_val_loss:.6f}）")

        if stopper.check(avg_val_loss):
            print("⛔ 早停触发，训练提前终止。")
            break

    return train_losses, val_losses
