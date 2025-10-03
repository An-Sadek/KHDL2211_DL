import os
import torch
import networkx as nx
import pandas as pd
from torch_geometric.utils import from_networkx, negative_sampling
from model import GAE, GCNEncoder

# ==========================
# HYPERPARAMETERS
# ==========================
DROPOUT = 0.5
BATCH_SIZE = 1024   # batch size for prediction
NUM_NEG_SAMPLES = 40000  # how many negative edges to sample
OUTPUT_CSV = "negative_predictions.csv"


# ==========================
# Load model and data
# ==========================
def load_model_and_data(file_path, model_path):
    # Load graph
    G = nx.read_edgelist(file_path, nodetype=int)
    data = from_networkx(G)
    data.x = torch.eye(data.num_nodes)  # Node features: identity matrix

    # Device setup
    device = torch.device("cpu")

    # Load model
    model = GAE(GCNEncoder(data.x.size(1), dropout=DROPOUT)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, data, device


# ==========================
# Predict negative edges
# ==========================
def predict_negative_edges(model, data, device, num_neg_samples=1000, batch_size=1024):
    model.eval()
    results = []

    with torch.no_grad():
        # Node embeddings
        z = model.encoder(data.x.to(device), data.edge_index.to(device))

        # Sample negative edges
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=num_neg_samples
        ).to(device)

        # Process in batches
        num_edges = neg_edge_index.size(1)
        for start in range(0, num_edges, batch_size):
            end = min(start + batch_size, num_edges)
            batch_edges = neg_edge_index[:, start:end]

            scores = model.decode(z, batch_edges)
            probs = torch.sigmoid(scores).cpu().numpy()

            edges = batch_edges.t().cpu().numpy()
            for (u, v), p in zip(edges, probs):
                results.append((u, v, p))

    return results


# ==========================
# Save results to CSV
# ==========================
def save_to_csv(results, output_file):
    df = pd.DataFrame(results, columns=["node_u", "node_v", "probability"])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} predictions to {output_file}")


# ==========================
# Main
# ==========================
def main():
    file_path = "facebook_combined.txt"
    model_path = "model_dict.pt"

    assert os.path.exists(file_path), f"File not found: {file_path}"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # Load model and data
    model, data, device = load_model_and_data(file_path, model_path)

    # Predict
    results = predict_negative_edges(model, data, device,
                                     num_neg_samples=NUM_NEG_SAMPLES,
                                     batch_size=BATCH_SIZE)

    # Save to CSV
    save_to_csv(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
