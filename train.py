import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, weight_decay=1e-5, patience=20):
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)

            y = batch.y
            if y.ndim == 1:
                y = y.view(output.shape)

            loss = criterion(output, y)

            # ✅ 检查 NaN，跳过这个 batch
            if torch.isnan(loss):
                print("⚠️ 检测到 NaN loss，跳过该 batch")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证集评估
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                y = batch.y
                if y.ndim == 1:
                    y = y.view(output.shape)

                val_loss = criterion(output, y)
                if torch.isnan(val_loss):
                    continue
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"📉 Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔ 提前停止训练于 epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses
