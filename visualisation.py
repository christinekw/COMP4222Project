import networkx as nx
import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.interpretability.saliency_maps import IntegratedGradients
import torch





# TODO: visualisation 
# 1. Node Importance Visualization Using Gradient-based Methods(Backpropagation)
# Gradients with Respect to Node Features: 
# In GCNs, the classification decision depends on the features of individual nodes 
# (which might represent residues) 
# and the edges between them (which represent the interactions between residues). 
# You can compute the gradient of the output class with respect to the input node features 
# (amino acid properties, for example) to find out which nodes most influence the final classification.
# Visualizing Important Nodes: For each node (residue), calculate how the gradient changes 
# the predicted output. Higher gradients indicate that the node’s feature is more important.
# Map these gradients onto the protein's 3D structure to see which residues are most important.
# Tools: PyTorch Geometric or DGL for implementing the GCN model, and PyMOL or Chimera to visualize residues in 3D.



#compute node importance for a single graph
#must separate from the training function as we use evaluation mode
def compute_node_importance(model, graph, target_class):
    model.eval()  # Set the model to evaluation mode
    
    # Make the input node features require gradients
    graph.x.requires_grad = True
    
    # Forward pass
    out = model(graph.x, graph.edge_index, graph.batch)
    target_score = out[:, target_class]  # Select score of the target class
    
    # Backward pass to compute gradients
    target_score.sum().backward()
    
    # Calculate the importance as the absolute gradient sum over features
    node_importance = graph.x.grad.abs().sum(dim=1)
    return node_importance

