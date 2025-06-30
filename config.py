# config.py

dataset = {
    "path": "IR_database_full.csv",
    "use_smooth": False,
    "clip_max": None
}

model = {
    "hidden_dim": 128,
    "output_dim": 3300,  # 标签维度与 crop_range 对应
    "num_layers": 5
}

train = {
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 300
}
