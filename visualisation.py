import torch
import dgl
import matplotlib.pyplot as plt
import networkx as nx
import json
from dgl.data import LegacyTUDataset
import torch.nn.functional as F
from model import HGPSLModel
from pyvis.network import Network
import random
import webbrowser
import os

# Load the model and configuration
def load_model_and_config(model_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    hyperparams = config["hyper-parameters"]
    
    model = HGPSLModel(
        in_feat=hyperparams["num_feature"], 
        out_feat=hyperparams["num_classes"],  
        hid_feat=hyperparams["hid_dim"],      
        dropout=hyperparams["dropout"],       
        pool_ratio=hyperparams["pool_ratio"], 
        conv_layers=hyperparams["conv_layers"],  
        pool_layers=hyperparams["pool_layers"]                     
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    
    return model, hyperparams, config

def sample_graph(dataset, split_ratios=(0.8, 0.1, 0.1)):
    num_training = int(len(dataset) * split_ratios[0])
    num_val = int(len(dataset) * split_ratios[1])
    num_test = len(dataset) - num_val - num_training
    
    test_graphs = dataset.graph_lists[num_training + num_val:]
    
    test_graph = random.choice(test_graphs)
    
    return test_graph

def compute_node_importance(model, graph, target_class):
    model.eval()  
    
    graph.ndata['feat'].requires_grad = True
    
    out = model(graph, graph.ndata['feat'])
    target_score = out[:, target_class]  
    
    target_score.sum().backward()
    
    node_importance = graph.ndata['feat'].grad.abs().sum(dim=1)
    return node_importance

def visualize_with_pyvis(graph, node_importance):
    nx_graph = dgl.to_networkx(graph).to_directed()
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    norm_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())

    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(nx_graph)

    for i, node in enumerate(net.nodes):
        importance_value = norm_importance[i].item()
        color_intensity = int(255 * (1 - importance_value))  
        color = f"rgb({255}, {color_intensity}, {color_intensity})"  
        node["color"] = color
        node["size"] = 10 + (importance_value * 40)  

    # Save the HTML file
    output_file = "node_importance_graph.html"
    net.show(output_file)
    
    # Open the generated HTML file automatically in the default browser
    webbrowser.open('file://' + os.path.realpath(output_file))

# Main function call example
if __name__ == '__main__':
    model_path = r"model\2conv1pool\best_model_cov2pool1_testaccuracy0.8393.pth"
    config_file = r"model\2conv1pool\Hidden=128_Pool=0.7_WeightDecay=0.001_Lr=0.001_Sample=True.log"
    model, hyperparams, config = load_model_and_config(model_path, config_file)
    dataset = LegacyTUDataset("PROTEINS")
    test_graph = sample_graph(dataset)
    node_importance = compute_node_importance(model, test_graph, target_class=1)
    visualize_with_pyvis(test_graph, node_importance)
