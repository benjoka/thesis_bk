import plotly.graph_objects as go
import plotly.colors as pc
from sklearn.metrics import adjusted_rand_score, rand_score
from gentrain.graph import (
    mean_edge_weight,
    max_edge_weight,
    build_mst,
    build_graph,
    export_graph_gexf,
    get_outbreak_community_labels
)
import community as community_louvain
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import random

def get_infection_rate(distance_matrix, distance_threshold = 1):
    tri_mask = np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)
    distance_matrix_df = pd.DataFrame(distance_matrix)
    filtered = distance_matrix_df.where(tri_mask).to_numpy()
    infections_count = (filtered <= distance_threshold).sum().sum()
    total_distances_count = np.count_nonzero(~np.isnan(filtered))
    return infections_count/total_distances_count

def get_infection_chain_participation_rate(distance_matrix, distance_threshold = 1):
    distance_matrix_df = pd.DataFrame(distance_matrix)
    mask_off_diagonal = ~np.eye(distance_matrix_df.shape[0], dtype=bool)
    distance_matrix_df = distance_matrix_df.where(mask_off_diagonal)
    condition = (distance_matrix_df < distance_threshold).any(axis=1)
    return condition.sum()/len(distance_matrix_df)
    
def get_infection_detection_scores(complete_matrix, observed_matrix, distance_threshold = 1):
    tri_mask = np.triu(np.ones(complete_matrix.shape), k=1).astype(bool)
    complete_matrix_df = pd.DataFrame(complete_matrix)
    filtered_complete_matrix = complete_matrix_df.where(tri_mask).to_numpy()
    observed_matrix_df = pd.DataFrame(observed_matrix)
    filtered_observed_matrix = observed_matrix_df.where(tri_mask).to_numpy()
    tp_mask = (filtered_observed_matrix <= distance_threshold) & (filtered_complete_matrix <= distance_threshold)
    tp = len(np.argwhere(tp_mask))
    tn_mask = (filtered_observed_matrix > distance_threshold) & (filtered_complete_matrix > distance_threshold)
    tn = len(np.argwhere(tn_mask))
    fp_mask = (filtered_observed_matrix <= distance_threshold) & (filtered_complete_matrix > distance_threshold)
    fp = len(np.argwhere(fp_mask))
    fn_mask = (filtered_observed_matrix > distance_threshold) & (filtered_complete_matrix <= distance_threshold)
    fn = len(np.argwhere(fn_mask))
    return tp, tn, fp, fn

def get_lineage_purity(lineages, communities):
    df = pd.DataFrame({
        "community": communities,
        "lineage":   lineages
    })
    counts = df.groupby(["community", "lineage"]) \
               .size() \
               .rename("count")
    community_sizes = counts.groupby(level=0) \
                            .sum() \
                            .rename("size")
    max_in_community = counts.groupby(level=0) \
                             .max() \
                             .rename("max_count")

    valid = community_sizes > 1
    community_sizes = community_sizes[valid]
    max_in_community = max_in_community[valid]
    
    return max_in_community.sum() / community_sizes.sum()
    
def candidate_evaluation_and_matrices(candidates, distance_matrix, runtime):
    distance_matrix_df = pd.DataFrame(distance_matrix)
    tri_mask = np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)
    filtered = distance_matrix_df.where(tri_mask)
    infections_count = (filtered < 2).sum().sum()
    total_distances_count = filtered.count().sum()
    mask = np.zeros_like(distance_matrix, dtype=bool)
    for i, candidates in candidates.items():
        for j in candidates:
            mask[i, j] = True
    tri_filtered = distance_matrix_df.where(tri_mask)
    candidate_matrix = pd.DataFrame(tri_filtered).where(mask).to_numpy()
    calculations = np.count_nonzero(~np.isnan(candidate_matrix))
    found_infections = (candidate_matrix < 2).sum().sum()
    tp, tn, fp, fn = get_infection_detection_scores(distance_matrix_df.to_numpy(), candidate_matrix)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2*tp) /((2*tp)+fp+fn)
    return {
        "computation_rate": calculations / total_distances_count,
        "infection_detection_rate": found_infections / infections_count,
        "infection_recall": recall,
        "infection_precision": precision,
        "infection_f1": found_infections / infections_count,
        "runtime": runtime,
    }, candidate_matrix


def get_community_map(partition):
    community_map = {}
    for sample_id, community_id in partition.items():
        if community_id not in community_map:
            community_map[community_id] = []
        community_map[community_id].append(sample_id)
    return community_map


def get_communities_larger_than(community_map, min_size):
    return {
        k: v
        for k, v in community_map.items()
        if isinstance(v, list) and len(v) > min_size
    }


def get_community_labels_for_sample_ids(community_labels, ids):
    return [community_labels[i] for i in ids]


def candidate_graph_evaluation(candidate_mst, complete_mst, candidate_community_labels, complete_community_labels, lineages):
    return {
        "mean_edge_weight": mean_edge_weight(candidate_mst),
        "mean_edge_weight_diff": mean_edge_weight(candidate_mst) - mean_edge_weight(complete_mst),
        "max_edge_weight": max_edge_weight(candidate_mst),
        "subgraph_count": len(list(nx.connected_components(candidate_mst))),
        "adjusted_rand_index": adjusted_rand_score(
            complete_community_labels, candidate_community_labels
        ),
        "lineage_purity": get_lineage_purity(lineages, candidate_community_labels),
        "lineage_purity_diff": get_lineage_purity(lineages, candidate_community_labels) - get_lineage_purity(lineages, complete_community_labels)
    }


def get_candidate_evaluation_and_export_mst(
    method_name,
    candidates,
    graph_path,
    distance_matrix,
    complete_community_labels,
    complete_mst,
    lineages,
    isolates_df,
    runtime,
):
    evaluation, matrix = candidate_evaluation_and_matrices(
        candidates, distance_matrix, runtime
    )
    candidate_graph = build_graph(matrix)
    candidate_mst = build_mst(candidate_graph)
    candidate_community_labels = get_outbreak_community_labels(candidate_mst)
    evaluation = evaluation | candidate_graph_evaluation(
        candidate_mst, complete_mst, candidate_community_labels, complete_community_labels, lineages
    )
    export_graph_gexf(
        candidate_mst,
        complete_community_labels,
        isolates_df,
        f"{graph_path}/{method_name}_{round(evaluation["computation_rate"], 2)}",
    )
    return evaluation

def get_computation_rate_plot(field, evaluation_collection, y_axis_title, legend=dict(
            x=0.55,
            y=0.1,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=35),
        ), computation_rates=[0.05,0.1,0.15,0.2]):
    fig = go.Figure()
    for index, method_evaluation in enumerate(evaluation_collection.items()):
        method = method_evaluation[0]
        evaluation_vs_computation_rate = method_evaluation[1]["values"]
        fig.add_trace(
            go.Scatter(
                x=list(evaluation_vs_computation_rate.keys()),
                y=[
                    values_vs_computation_rate[field]
                    for values_vs_computation_rate in evaluation_vs_computation_rate.values()
                ],
                mode="lines",
                name=f"{method}",
                line=dict(
                    width=4,
                    dash=method_evaluation[1]["stroke"],
                    color=method_evaluation[1]["color"],
                ),
            )
        )
        for (
            computation_rate,
            evaluation_vs_computation_rate,
        ) in evaluation_vs_computation_rate.items():
            fig.add_trace(
                go.Scatter(
                    text="",
                    x=[computation_rate],
                    y=[evaluation_vs_computation_rate[field]],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color=method_evaluation[1]["color"],
                    ),
                    showlegend=False,
                )
            )
    fig.update_layout(
        width=1200,
        height=700,
        xaxis_title="Computation rate",
        yaxis_title=y_axis_title,   
        margin=dict(
            l=110,
            r=0,
            t=0,
            b=80  
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=computation_rates
        ),
        legend=legend,
        font=dict(size=25),
        template="presentation",
    )
    return fig
