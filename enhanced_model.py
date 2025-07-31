import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d, Dropout

class EnhancedSyntaxGCN(torch.nn.Module):
    def __init__(self, num_node_types, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        
        # Input layer
        self.conv1 = GCNConv(3, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        
        # Hidden layers
        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_layers.append(BatchNorm1d(hidden_dim))
        
        # Output layers
        self.dropout = Dropout(dropout)
        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for concatenated pooling
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, 2)  # Binary classification
        
        # Batch normalization for final layers
        self.bn_final1 = BatchNorm1d(hidden_dim)
        self.bn_final2 = BatchNorm1d(hidden_dim // 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling (concatenate mean and max pooling)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Final classification layers
        x = self.lin1(x)
        x = self.bn_final1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = self.bn_final2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin3(x)
        return x

class AttentionSyntaxGCN(torch.nn.Module):
    def __init__(self, num_node_types, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Multi-head attention mechanism
        self.attention_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = torch.nn.Linear(3, hidden_dim)
        
        # Attention layers
        self.attention_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        
        for _ in range(2):
            self.attention_layers.append(
                torch.nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.norm_layers.append(torch.nn.LayerNorm(hidden_dim))
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        
        # Output layers
        self.dropout = Dropout(dropout)
        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, 2)
        
        self.bn_final1 = BatchNorm1d(hidden_dim)
        self.bn_final2 = BatchNorm1d(hidden_dim // 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Final classification layers
        x = self.lin1(x)
        x = self.bn_final1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = self.bn_final2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin3(x)
        return x

class ResidualSyntaxGCN(torch.nn.Module):
    def __init__(self, num_node_types, hidden_dim=64, num_layers=4, dropout=0.3):
        super().__init__()
        
        # Input layer
        self.input_proj = torch.nn.Linear(3, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            block = ResidualBlock(hidden_dim, dropout)
            self.residual_blocks.append(block)
        
        # Output layers
        self.dropout = Dropout(dropout)
        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin3 = torch.nn.Linear(hidden_dim // 2, 2)
        
        self.bn_final1 = BatchNorm1d(hidden_dim)
        self.bn_final2 = BatchNorm1d(hidden_dim // 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x, edge_index)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Final classification layers
        x = self.lin1(x)
        x = self.bn_final1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = self.bn_final2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin3(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        identity = x
        
        # First convolution
        out = self.conv1(x, edge_index)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out, edge_index)
        out = self.bn2(out)
        
        # Residual connection
        out = out + identity
        out = F.relu(out)
        
        return out

# Keep the original model for backward compatibility
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