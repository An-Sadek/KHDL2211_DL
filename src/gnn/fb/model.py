import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2048)
        self.conv2 = GCNConv(2048, 1024)
        self.conv3 = GCNConv(1024, 512)
        self.conv4 = GCNConv(512, 256)
        self.conv5 = GCNConv(256, 128)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv5(x, edge_index)
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
        assert z.dim() == 2, f"Expected 2D node embeddings, got shape {z.shape}"
        product = z[edge_index[0]] * z[edge_index[1]]
        return product.sum(dim=1)


if __name__ == "__main__":
    pass