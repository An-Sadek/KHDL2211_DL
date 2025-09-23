import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, train_test_split_edges, negative_sampling
import networkx as nx
import os

from model import GAE, GCNEncoder


# Load graph
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

# Add feature
data.x = torch.eye(data.num_nodes)  # Or use torch.randn(data.num_nodes, 16) for smaller features
data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.1)

# Load model & prepare
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

# Train
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    
    # Explicitly compute node embeddings
    z = model.encoder(x, train_pos_edge_index)  # Shape: (4039, 64)
    
    # Sample negative edges
    neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_pos_edge_index.size(1),
    ).to(device)

    # Get scores for positive and negative edges
    pos_out = model.decode(z, train_pos_edge_index)  # Shape: (num_edges,)
    neg_out = model.decode(z, neg_edge_index)  # Shape: (num_edges,)
    out = torch.cat([pos_out, neg_out], dim=0)

    labels = get_link_labels(train_pos_edge_index, neg_edge_index).to(device)
    loss = F.binary_cross_entropy_with_logits(out, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# Save model
model.eval()
z = model.encoder(x, train_pos_edge_index)
print("Node embeddings shape:", z.shape)
torch.save(model.state_dict(), "model_dict.pt")