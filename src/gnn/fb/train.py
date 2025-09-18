import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx, train_test_split_edges
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
data = from_networkx(G)

# Add identity features (if none exist)
data.x = torch.eye(data.num_nodes)

# Split edges for link prediction
data = train_test_split_edges(data)

# --------------------------
# 2. Define GAE Model
# --------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GAE(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        # inner product decoder
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def decode_all(self, z, threshold=0.5):
        prob_adj = torch.sigmoid(torch.matmul(z, z.t()))
        return (prob_adj > threshold).nonzero(as_tuple=False).t()

# --------------------------
# 3. Train
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GAE(GCNEncoder(data.x.size(1), 64)).to(device)
x, train_pos_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def get_link_labels(pos_edge_index, neg_edge_index):
    num_pos = pos_edge_index.size(1)
    num_neg = neg_edge_index.size(1)
    labels = torch.zeros(num_pos + num_neg, dtype=torch.float)
    labels[:num_pos] = 1.
    return labels

for epoch in range(1, 11):
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index)

    # Sample negative edges
    from torch_geometric.utils import negative_sampling
    neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
    )

    # Get scores
    pos_out = model.decode(z, train_pos_edge_index)
    neg_out = model.decode(z, neg_edge_index)
    out = torch.cat([pos_out, neg_out], dim=0)

    labels = get_link_labels(train_pos_edge_index, neg_edge_index).to(device)

    loss = F.binary_cross_entropy_with_logits(out, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# --------------------------
# 4. Inference
# --------------------------
model.eval()
z = model(x, train_pos_edge_index)
print("Node embeddings shape:", z.shape)
