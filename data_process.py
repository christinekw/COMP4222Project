import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch

def average_size_protein(dataset):
    graph_sizes = [graph.number_of_nodes() for graph, label in dataset]
    return sum(graph_sizes) / len(graph_sizes)
    
def select_subset_sizecriteria(dataset,size):
    # size (string) : "long" or "short"
    threshold = average_size_protein(dataset)
    if size == "long":
        filtered_graphs = [G for G in graph_dataset if G.number_of_nodes() >= threshold]
    else if size == "short":
        filtered_graphs = [G for G in graph_dataset if G.number_of_nodes() < node_threshold]
    else:
        print("Error: invalid size parameter when selecting subset")
        return
    return filtered_graphs

