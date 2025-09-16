import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn import GCNConv

# --------------------------
# 1. Load graph
# --------------------------
def load_edge_list(file_path, directed=False):
    if directed:
        G = nx.read_edgelist(file_path, nodetype=int, create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(file_path, nodetype=int)
    return G

file_path = "data/facebook/facebook_combined.txt"
assert os.path.exists(file_path), "Can't find path"
G = load_edge_list(file_path)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# --------------------------
# 2. Convert to PyG Data object
# --------------------------
data = from_networkx(G)

num_nodes = G.number_of_nodes()
embedding_dim = 64
embedding = nn.Embedding(num_nodes, embedding_dim)

node_ids = torch.arange(num_nodes)
data.x = embedding(node_ids)  # node embeddings

# --------------------------
# 3. Define a simple GCN Encoder
# --------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# --------------------------
# 4. Link prediction model
# --------------------------
class LinkPredictor(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GCNEncoder(embedding_dim, hidden_channels)

    def decode(self, z, edge_index):
        # Dot product between node pairs
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)  # [num_edges]

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encoder(x, edge_index)
        pos_score = self.decode(z, pos_edge_index)
        neg_score = self.decode(z, neg_edge_index)
        return pos_score, neg_score

# --------------------------
# 5. Training setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinkPredictor(hidden_channels=32).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Positive edges = existing friendships
pos_edge_index = data.edge_index

for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()

    # Sample negative edges (non-friend pairs)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    )

    pos_score, neg_score = model(data.x, data.edge_index, pos_edge_index, neg_edge_index)

    # Labels: 1 for positive edges, 0 for negative edges
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    scores = torch.cat([pos_score, neg_score])

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward(retain_graph=True)
    optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            pred = torch.sigmoid(scores) > 0.5
            acc = (pred == labels.bool()).float().mean().item()
            print(f"Epoch {epoch:03d}, Loss {loss:.4f}, Acc {acc:.4f}")
