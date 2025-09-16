import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx

# Example: build Data object from ego-net
G = nx.read_edgelist("data/facebook/data/0.edges")
# Map nodes to indices
mapping = {n: i for i, n in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)

edge_index = torch.tensor(list(G.edges())).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected

# Fake features: identity matrix if no .feat file
x = torch.eye(G.number_of_nodes())

# Labels: here you’d load from .circles file
y = torch.randint(0, 2, (G.number_of_nodes(),))

data = Data(x=x, edge_index=edge_index, y=y)

# Simple GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_channels=data.num_features,
            hidden_channels=16,
            out_channels=int(data.y.max().item())+1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
