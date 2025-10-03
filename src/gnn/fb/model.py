import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 256)
        self.conv3 = GCNConv(256, 128)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GAE(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        adj_pred = self.decode(z, edge_index)
        return adj_pred

    def decode(self, z, edge_index):
        # Ensure z is 2D (node embeddings)
        product = z[edge_index[0]] * z[edge_index[1]]
        return product.sum(dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Example input channels (adjust based on your data)
    in_channels = 4039
    encoder = GCNEncoder(in_channels=in_channels)
    model = GAE(encoder)
    
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params}")