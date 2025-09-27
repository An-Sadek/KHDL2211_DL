import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from model import GAE, GCNEncoder

# HYPERPARAMETERS
DROPOUT = 0.5

def load_model_and_data(file_path, model_path):
    # Load graph
    G = nx.read_edgelist(file_path, nodetype=int)
    data = from_networkx(G)
    data.x = torch.eye(data.num_nodes)  # Node features: identity matrix
    
    # Device setup
    device = torch.device("cpu")  # Explicitly use CPU
    
    # Load model
    model = GAE(GCNEncoder(data.x.size(1), dropout=DROPOUT)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, data, device

def predict_link_probability(model, data, device, node_id0, node_id1):
    model.eval()
    with torch.no_grad():
        # Generate node embeddings
        z = model.encoder(data.x.to(device), data.edge_index.to(device))
        
        # Create edge tensor for the specific node pair
        edge = torch.tensor([[node_id0, node_id1]], dtype=torch.long).t().to(device)
        
        # Predict score for the edge
        score = model.decode(z, edge)
        probability = torch.sigmoid(score).item()
        
    return probability

def main(node_id0, node_id1):
    file_path = "facebook_combined.txt"
    model_path = "model_dict.pt"
    
    # Load model and data
    model, data, device = load_model_and_data(file_path, model_path)
    
    # Predict link probability
    probability = predict_link_probability(model, data, device, node_id0, node_id1)
    
    return probability

if __name__ == "__main__":
    # Example usage: replace with desired node IDs
    node_id0, node_id1 = 0, 2
    prob = main(node_id0, node_id1)
    print(f"Predicted link probability between node {node_id0} and node {node_id1}: {prob:.4f}")