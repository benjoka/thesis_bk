import numpy as np
import networkx as nx
import pandas as pd
import community as community_louvain
import time 

def get_outbreak_community_labels(graph, distance_threshold = None):
    graph_copy = graph.copy()
    if distance_threshold:
        edges_to_remove = [(u, v) for u, v, d in graph_copy.edges(data=True) if d['weight'] > distance_threshold]
        graph_copy.remove_edges_from(edges_to_remove)
    partition = community_louvain.best_partition(graph_copy, random_state=42)
    community_labels = list(partition.values())
    return community_labels

def get_connected_component_labels(mst, distance_threshold = 1):
    mst_copy = mst.copy()
    edges_to_remove = [(u, v) for u, v, d in mst_copy.edges(data=True) if d['weight'] > distance_threshold]
    mst_copy.remove_edges_from(edges_to_remove)
    communities = list(nx.connected_components(mst_copy))
    outbreak_communities = sorted(communities, key=len, reverse=True)
    community_labels = {}
    for community_id, community_nodes in enumerate(outbreak_communities):
        for node_id in community_nodes:
            community_labels[node_id] = community_id
    community_labels = dict(sorted(community_labels.items()))
    community_labels = [label for label in community_labels.values()]
    return community_labels
    
def mean_edge_weight(G):
    total_weight = sum(data.get("weight") for u, v, data in G.edges(data=True))
    num_edges = G.number_of_edges()
    return total_weight / num_edges if num_edges > 0 else 0.0

def median_edge_weight(G):
    weights = [data.get("weight") for u, v, data in G.edges(data=True)]
    if not weights:
        return 0.0
    return np.median(weights)

def max_edge_weight(G):
    return max(data.get("weight") for u, v, data in G.edges(data=True))

def build_mst(graph):
    start = time.time()
    mst = nx.minimum_spanning_tree(graph, algorithm="prim")
    for _, _, data in mst.edges(data=True):
        # since gephi can't deal with weights of 0, we set 0 weights to 0.1
        # this does not effect resulting graphs since all other weights are interger values
        # and 0.1 remaind the smallest present edge weight
        if data.get("weight", 1) == 0:
            data["weight"] = 0.1   
    end = time.time()
    print(f"mst generation time: {round(end - start, 2)}s")
    return mst

def build_graph(distance_matrix):
    graph = nx.Graph()
    n = distance_matrix.shape[0]
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            distance = distance_matrix[i][j]
            if not np.isnan(distance):
                label = "regular"
                if distance < 2:
                    label = "infection"
                elif distance > 5:
                    label = "outlier"
                graph.add_edge(i, j, weight=distance, color=label)
    return graph
    
def export_graph_gexf(graph, community_labels, dataset, name):
    datetime_sampling_dates = pd.to_datetime(dataset["date_of_sampling"])
    numeric_dates = (datetime_sampling_dates - datetime_sampling_dates.min()).dt.days
    nx.set_node_attributes(graph, {node: community_labels[int(node)] for node in graph.nodes()}, name="community")
    nx.set_node_attributes(graph, {node: dataset.iloc[int(node)]["clade"] for node in graph.nodes()}, name="clade")            
    nx.set_node_attributes(graph, {node: dataset.iloc[int(node)]["Nextclade_pango"] for node in graph.nodes()}, name="pango")
    nx.set_node_attributes(graph, {node: dataset.iloc[int(node)]["prime_diagnostic_lab.city"] for node in graph.nodes()}, name="city")
    nx.set_node_attributes(graph, {node: dataset.iloc[int(node)]["prime_diagnostic_lab.state"] for node in graph.nodes()}, name="state")
    nx.set_node_attributes(graph, {node: numeric_dates.iloc[int(node)] for node in graph.nodes()}, name="sampling_date")
    nx.write_gexf(graph, f"{name}.gexf")