import networkx as nx
import matplotlib.pyplot as plt

def generate_graph(distance_matrix):
    graph = nx.Graph()
    fasta_ids = list(distance_matrix.keys())
    edges = []
    for index_1, fasta_id_1 in enumerate(distance_matrix):
        for fasta_id_2 in distance_matrix[fasta_id_1]:
            edges.append((fasta_ids[index_1], distance_matrix[fasta_id_1][fasta_id_2], {"weight": distance_matrix[fasta_id_1][fasta_id_2]}))

    graph.add_edges_from(
        edges
    )
    return graph

def generate_mst(graph):
    mst = nx.minimum_spanning_tree(graph)
    return mst

def visualize_mst(mst):
    plt.figure(3, figsize=(10, 10))
    pos = nx.spring_layout(mst, k=0.5, iterations=200, weight="distance")
    nx.draw_networkx_nodes(mst, pos, node_color="lightblue", node_size=100)
    nx.draw_networkx_labels(mst, pos, font_size=10, font_family="sans-serif")
    nx.draw_networkx_edge_labels(
        mst, pos, edge_labels={(u, v): d["weight"] for u, v, d in mst.edges(data=True)}
    )
    nx.draw_networkx_edges(mst, pos, edge_color="green", width=1)
    plt.axis("off")
    plt.show()
