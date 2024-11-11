import torch
import torch.nn
import torch.nn.functional as F

from dgl.nn import AvgPooling, MaxPooling
from layers import ConvPoolReadout


# TODO: may be try to train the model with 1 pooling, 2 pooling. And then compare to the reported one using 3 pooling layers.
# Then, compare to a baseline model: single layer GCN

# TODO:Select subsets of proteins from the database according to
# 1. size
#   Split the test data based on protein size (e.g., short, medium, and long sequences).
#   You could define these categories based on quantiles of sequence length in your dataset 
#   or based on biologically relevant thresholds.
# 2. structure
#   Curate test sets that represent various structural classes, such as alpha, beta, and mixed alpha/beta structures, or different protein folds (e.g., all-alpha, all-beta, alpha-beta).
# Then, measure Performance Across Groups
# Finally, investigate the robustness of a model based on different protein sizes and structures
# e.g.  Identify if there’s a correlation between sequence length and model performance. 
# e.g. Evaluate performance on structurally distinct test sets to see if there are any biases.

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
class HGPSLModel(torch.nn.Module):
    r"""

    Description
    -----------
    The graph classification model using HGP-SL pooling.

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels.
    out_feat : int
        The number of output node feature's channels.
    hid_feat : int
        The number of hidden state's channels.
    dropout : float, optional
        The dropout rate. Default: 0
    pool_ratio : float, optional
        The pooling ratio for each pooling layer. Default: 0.5
    conv_layers : int, optional
        The number of graph convolution and pooling layers. Default: 3
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency.
        Currently we only support full graph. Default: :obj:`False`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    """

    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        hid_feat: int,
        dropout: float = 0.0,
        pool_ratio: float = 0.5,
        conv_layers: int = 3,
        sample: bool = False,
        sparse: bool = True,
        sl: bool = True,
        lamb: float = 1.0,
    ):
        super(HGPSLModel, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hid_feat = hid_feat
        self.dropout = dropout
        self.num_layers = conv_layers
        self.pool_ratio = pool_ratio

        convpools = []
        for i in range(conv_layers):
            c_in = in_feat if i == 0 else hid_feat
            c_out = hid_feat
            use_pool = i != conv_layers - 1
            convpools.append(
                ConvPoolReadout(
                    c_in,
                    c_out,
                    pool_ratio=pool_ratio,
                    sample=sample,
                    sparse=sparse,
                    sl=sl,
                    lamb=lamb,
                    pool=use_pool,
                )
            )
        self.convpool_layers = torch.nn.ModuleList(convpools)

        self.lin1 = torch.nn.Linear(hid_feat * 2, hid_feat)
        self.lin2 = torch.nn.Linear(hid_feat, hid_feat // 2)
        self.lin3 = torch.nn.Linear(hid_feat // 2, self.out_feat)

    def forward(self, graph, n_feat):
        final_readout = None
        e_feat = None

        for i in range(self.num_layers):
            graph, n_feat, e_feat, readout = self.convpool_layers[i](
                graph, n_feat, e_feat
            )
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        n_feat = F.relu(self.lin1(final_readout))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = F.relu(self.lin2(n_feat))
        n_feat = F.dropout(n_feat, p=self.dropout, training=self.training)
        n_feat = self.lin3(n_feat)

        return F.log_softmax(n_feat, dim=-1)
