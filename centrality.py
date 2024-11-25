import argparse
import visualisation
import networkx as nx
import dgl
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    # model_path = r"model\2conv1pool\best_model_cov2pool1_testaccuracy0.8393.pth"
    # config_file = r"model\2conv1pool\Hidden=128_Pool=0.7_WeightDecay=0.001_Lr=0.001_Sample=True.log"
    parser = argparse.ArgumentParser(description="Load model and config for graph processing")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model file (.pth)")
    parser.add_argument('--config_file', type=str, required=True, help="Path to the configuration file (.log)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with provided arguments
    G,node_importance = visualisation.anotherversion(args.model_path, args.config_file)
    norm_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())
    nx_graph = dgl.to_networkx(G)
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(nx_graph)

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(nx_graph)

    # Print the results
    # print("Betweenness Centrality:")
    bc = []
    for node, centrality in betweenness_centrality.items():
        bc.append(centrality)
    print(len(bc))
    dc=[]
    # print("\nDegree Centrality:")
    for node, centrality in degree_centrality.items():
        dc.append(centrality)
    print(len(dc))
    
    nodes_importance =[]
    for i in range(len(norm_importance)):
        nodes_importance.append(norm_importance[i].item())
    
    # Data for Betweenness Centrality, Degree Centrality, and Node Importance
    nodes = np.arange(len(bc))


    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(nodes, bc, label='Betweenness Centrality', marker='o', color='r')
    plt.plot(nodes, dc, label='Degree Centrality', marker='s', color='b')
    plt.plot(nodes, nodes_importance, label='Node Importance', marker='^', color='g')

    plt.title('Betweenness Centrality, Degree Centrality, and Node Importance')
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
