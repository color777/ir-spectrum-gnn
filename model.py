import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GINConv,
    GCNConv,
    JumpingKnowledge,
    global_mean_pool
)

# ===== BaseBlock 可复用 MLP Layer =====
class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.residual = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.mlp(x) + self.residual(x)

# ===== GATv2 + JK =====
class GATv2_JK_Model(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_layers=5, heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.jump = JumpingKnowledge(mode="lstm", channels=hidden_dim, num_layers=num_layers)
        self.pool = global_mean_pool
        self.mlp = MLPBlock(hidden_dim, output_dim, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        xs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            xs.append(x)
        x = self.jump(xs)
        x = self.pool(x, batch)
        return self.mlp(x)

# ===== GIN + JK =====
class GIN_JK_Model(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_layers=5, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_mlp))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.jump = JumpingKnowledge(mode="lstm", channels=hidden_dim, num_layers=num_layers)
        self.pool = global_mean_pool
        self.mlp = MLPBlock(hidden_dim, output_dim, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        xs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            xs.append(x)
        x = self.jump(xs)
        x = self.pool(x, batch)
        return self.mlp(x)

# ===== GCN + JK =====
class GCN_JK_Model(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_layers=5, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.jump = JumpingKnowledge(mode="lstm", channels=hidden_dim, num_layers=num_layers)
        self.pool = global_mean_pool
        self.mlp = MLPBlock(hidden_dim, output_dim, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        xs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            xs.append(x)
        x = self.jump(xs)
        x = self.pool(x, batch)
        return self.mlp(x)
