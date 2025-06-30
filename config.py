# config.py

dataset = {
    "path": "IR_database_full.csv",
    "use_smooth": True,
    "clip_max": 1.0
}

model = {
    "hidden_dim": 128,
    "output_dim": None,
    "num_layers": 3
}

train = {
    "epochs": 300,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "ensemble": 3  # 模型集成数量
}
