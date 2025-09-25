import os
import torch
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from torch_geometric.utils import from_networkx, train_test_split_edges, negative_sampling

from model import GAE, GCNEncoder

# -----------------------------
# Load graph
# -----------------------------
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

# Node features: identity matrix (one-hot per node)
data.x = torch.eye(data.num_nodes)

# Train/val/test split
data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.1)

# -----------------------------
# Model setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAE(GCNEncoder(data.x.size(1), dropout=0.25)).to(device)
x, train_pos_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Helper functions
# -----------------------------
def get_link_labels(pos_edge_index, neg_edge_index):
    num_pos = pos_edge_index.size(1)
    num_neg = neg_edge_index.size(1)
    labels = torch.zeros(num_pos + num_neg, dtype=torch.float)
    labels[:num_pos] = 1.
    return labels

def evaluate_hits_mrr(model, data, pos_edge_index, neg_edge_index, k_values=[20, 50, 100]):
    model.eval()
    with torch.no_grad():
        z = model.encoder(data.x.to(device), data.train_pos_edge_index.to(device))
        hits = {k: 0 for k in k_values}
        mrr_sum = 0
        num_tests = pos_edge_index.size(1)

        for i in range(num_tests):
            pos_edge = pos_edge_index[:, i:i+1]
            neg_edges = neg_edge_index[:, i*neg_edge_index.size(1)//num_tests:(i+1)*neg_edge_index.size(1)//num_tests]
            edge_indices = torch.cat([pos_edge, neg_edges], dim=1).to(device)

            scores = model.decode(z, edge_indices)
            _, indices = torch.sort(scores, descending=True)
            rank = (indices == 0).nonzero(as_tuple=True)[0].item() + 1

            for k in k_values:
                if rank <= k:
                    hits[k] += 1
            mrr_sum += 1.0 / rank

        hits_results = {k: hits[k] / num_tests for k in k_values}
        mrr = mrr_sum / num_tests

    return hits_results, mrr

# -----------------------------
# Training loop
# -----------------------------
training_logs = []
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    z = model.encoder(x, train_pos_edge_index)
    neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_pos_edge_index.size(1),
    ).to(device)

    pos_out = model.decode(z, train_pos_edge_index)
    neg_out = model.decode(z, neg_edge_index)
    out = torch.cat([pos_out, neg_out], dim=0)

    labels = get_link_labels(train_pos_edge_index, neg_edge_index).to(device)
    loss = F.binary_cross_entropy_with_logits(out, labels)
    loss.backward()
    optimizer.step()

    # Evaluate on validation set every epoch
    val_neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.size(1) * 10,
    ).to(device)
    hits_results, mrr = evaluate_hits_mrr(model, data, data.val_pos_edge_index, val_neg_edge_index)

    # Log results
    log_entry = {
        'epoch': epoch,
        'loss': loss.item(),
        'hits@20': hits_results[20],
        'hits@50': hits_results[50],
        'hits@100': hits_results[100],
        'mrr': mrr
    }
    training_logs.append(log_entry)

    print(
        f"Epoch {epoch:02d}, Loss {loss.item():.4f}, "
        f"Hits@20: {hits_results[20]:.4f}, "
        f"Hits@50: {hits_results[50]:.4f}, "
        f"Hits@100: {hits_results[100]:.4f}, "
        f"MRR: {mrr:.4f}"
    )

# -----------------------------
# Final test evaluation
# -----------------------------
test_neg_edge_index = negative_sampling(
    edge_index=data.test_pos_edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=data.test_pos_edge_index.size(1) * 10,
).to(device)

hits_results, mrr = evaluate_hits_mrr(model, data, data.test_pos_edge_index, test_neg_edge_index)
print(
    f"Final Test - Hits@20: {hits_results[20]:.4f}, "
    f"Hits@50: {hits_results[50]:.4f}, "
    f"Hits@100: {hits_results[100]:.4f}, "
    f"MRR: {mrr:.4f}"
)

# -----------------------------
# Save logs & model
# -----------------------------
df = pd.DataFrame(training_logs)
df.to_csv('training_results.csv', index=False)
print("Training results saved to 'training_results.csv'")

model.eval()
z = model.encoder(x, train_pos_edge_index)
print("Node embeddings shape:", z.shape)
torch.save(model.state_dict(), "model_dict.pt")
