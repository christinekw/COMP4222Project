import warnings
warnings.filterwarnings("ignore")
import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch

def average_size_protein(dataset):
    graph_sizes = [graph.number_of_nodes() for graph, label in dataset]
    return sum(graph_sizes) / len(graph_sizes)
    
#return type has to be torch.utils.data.Subset
def select_subset_sizecriteria(dataset,testset,size):
    # size (string) : "long" or "short"
    threshold = average_size_protein(dataset)
    length = len(testset)
    mask =[]
    if size == "long":
        for j in range(length):
            if testset[j][0].num_nodes() >= threshold:
                mask.append(j)
        
    elif size == "short":
        for j in range(length):
            if testset[j][0].num_nodes() < threshold:
                mask.append(j)
    else:
        raise ValueError("Invalid size criteria")
    subset = torch.utils.data.Subset(testset, mask)
    return subset
