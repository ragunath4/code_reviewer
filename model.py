import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class SyntaxGCN(torch.nn.Module):
    def __init__(self, num_node_types, hidden_dim=32):
        super().__init__()
        # 3 input features: type idx, depth, num children
        self.conv1 = GCNConv(3, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x
