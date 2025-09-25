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

file_path = "data/facebook/data/facebook_combined.txt"
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
    labels = torch.zeros(num_pos + num_neg, dtype=torch.float, device=device)
    labels[:num_pos] = 1.
    return labels

def evaluate_mrr(model, data, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encoder(data.x.to(device), data.train_pos_edge_index.to(device))
        num_tests = pos_edge_index.size(1)
        mrr_sum = 0

        for i in range(num_tests):
            pos_edge = pos_edge_index[:, i:i+1].to(device)
            neg_edges = neg_edge_index[:, i*neg_edge_index.size(1)//num_tests:(i+1)*neg_edge_index.size(1)//num_tests].to(device)
            edge_indices = torch.cat([pos_edge, neg_edges], dim=1)

            scores = model.decode(z, edge_indices)
            _, indices = torch.sort(scores, descending=True)
            rank = (indices == 0).nonzero(as_tuple=True)[0].item() + 1
            mrr_sum += 1.0 / rank

        mrr = mrr_sum / num_tests
    return mrr

# -----------------------------
# Training loop
# -----------------------------
training_logs = []
num_epochs = 300

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    # -------- Train step --------
    z = model.encoder(x, train_pos_edge_index)
    neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_pos_edge_index.size(1),
    ).to(device)

    pos_out = model.decode(z, train_pos_edge_index)
    neg_out = model.decode(z, neg_edge_index)
    out = torch.cat([pos_out, neg_out], dim=0)

    labels = get_link_labels(train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(out, labels)
    loss.backward()
    optimizer.step()

    # -------- Validation step --------
    model.eval()
    with torch.no_grad():
        val_neg_edge_index = negative_sampling(
            edge_index=data.val_pos_edge_index.to(device),
            num_nodes=data.num_nodes,
            num_neg_samples=data.val_pos_edge_index.size(1) * 10,
        ).to(device)

        z = model.encoder(data.x.to(device), data.train_pos_edge_index.to(device))

        pos_val_out = model.decode(z, data.val_pos_edge_index.to(device))
        neg_val_out = model.decode(z, val_neg_edge_index)
        out_val = torch.cat([pos_val_out, neg_val_out], dim=0)

        labels_val = get_link_labels(data.val_pos_edge_index.to(device), val_neg_edge_index)
        loss_val = F.binary_cross_entropy_with_logits(out_val, labels_val)

        # Compute MRR for training set
        mrr_train = evaluate_mrr(model, data, train_pos_edge_index, neg_edge_index)

        # Compute MRR for validation set
        mrr_val = evaluate_mrr(model, data, data.val_pos_edge_index.to(device), val_neg_edge_index)

    # -------- Logging --------
    log_entry = {
        'epoch': epoch,
        'loss': loss.item(),
        'loss_val': loss_val.item(),
        'mrr': mrr_train,
        'mrr_val': mrr_val
    }
    training_logs.append(log_entry)

    print(
        f"Epoch {epoch:02d}, "
        f"Loss {loss.item():.4f}, Val Loss {loss_val.item():.4f}, "
        f"MRR (train): {mrr_train:.4f}, MRR (val): {mrr_val:.4f}"
    )

# -----------------------------
# Final test evaluation
# -----------------------------
test_neg_edge_index = negative_sampling(
    edge_index=data.test_pos_edge_index.to(device),
    num_nodes=data.num_nodes,
    num_neg_samples=data.test_pos_edge_index.size(1) * 10,
).to(device)

mrr_test = evaluate_mrr(model, data, data.test_pos_edge_index.to(device), test_neg_edge_index)
print(f"Final Test - MRR: {mrr_test:.4f}")

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