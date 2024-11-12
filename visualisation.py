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
# Map these gra