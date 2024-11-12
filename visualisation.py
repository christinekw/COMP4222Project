# TODO: visualisation 
# 1. Node Importance Visualization Using Gradient-based Methods
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

