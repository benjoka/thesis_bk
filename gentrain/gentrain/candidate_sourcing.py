import numpy as np
import random
import time
from gentrain.distance_matrix import get_bitwise_xor_distance_matrix
import hnswlib
from collections import defaultdict
from itertools import combinations

def get_hnsw_candidates(
    encodings, limit, print_execution_time=True
):
    start = time.time()
    num_elements = len(encodings)
    data = np.float32(np.array(list(encodings.values())))
    ids = np.arange(len(data))
    index = hnswlib.Index(space = "l2", dim = len(data[0]))
    index.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
    index.add_items(data, ids)
    index.set_ef(50)
    hnsw_candidates, hnsw_distances = index.knn_query(data, int(limit/num_elements))
    candidates = {}
    for sequence_index, sequence_candidates in enumerate(hnsw_candidates):
        candidates[sequence_index] = sequence_candidates
    end = time.time()
    
    runtime = round((end - start), 2)
    if print_execution_time:
        print(f"execution time {limit}: {runtime}s")
    return candidates, runtime

def breadth_bitwise_xor_candidates(vectors_dict, limit, index):
    candidates = {index: [] for index in vectors_dict.keys()}
    candidate_tuples_with_distances = {}
    distance_matrix = get_bitwise_xor_distance_matrix(vectors_dict)
    distance_collection_start = time.time()
    for vector_id, candidate_distances in enumerate(distance_matrix):
        for candidate_id, distance in enumerate(candidate_distances):
            if candidate_id == vector_id:
                continue
            candidate_tuples_with_distances[
                min(vector_id, candidate_id), max(vector_id, candidate_id)
            ] = distance
    distance_collection_end = time.time()
    print(f"execution time distance collection: {round((distance_collection_end - distance_collection_start), 2)}s")

    candidates_start = time.time()
    sorted_candidate_tuples_with_distances = dict(
        sorted(candidate_tuples_with_distances.items(), key=lambda item: item[1])
    )
    candidate_tuples = list(sorted_candidate_tuples_with_distances.keys())

    for vector_1, vector_2 in candidate_tuples:
        if len(candidates[vector_1]) <= int(limit / len(vectors_dict)):
            candidates[vector_1].append(vector_2)

    candidates_end = time.time()
    print(
        f"execution time breadth search: {round((candidates_end - candidates_start), 2)}s"
    )

    return candidates


def depth_bitwise_xor_candidates(vectors_dict, limit, index):
    candidates = {index: [] for index in vectors_dict.keys()}
    candidate_tuples_with_distances = {}
    distance_collection_start = time.time()
    packed_vectors = np.packbits(np.array(list(vectors_dict.values())), axis=1)
    for vector_id, encoding in vectors_dict.items():
        xor_result = np.bitwise_xor(packed_vectors, packed_vectors[vector_id])
        xor_distances = np.unpackbits(xor_result, axis=1).sum(axis=1)
        for candidate_id, xor_distance in enumerate(xor_distances):
            if candidate_id == vector_id:
                continue
            candidate_tuples_with_distances[
                    min(vector_id, candidate_id), max(vector_id, candidate_id)
                ] = xor_distance
    distance_collection_end = time.time()
    print(f"execution time xor distance calculation: {round((distance_collection_end - distance_collection_start), 2)}s")

    candidates_start = time.time()
    sorted_candidate_tuples_with_distances = dict(
        sorted(candidate_tuples_with_distances.items(), key=lambda item: item[1])
    )
    relevant_candidate_tuples = list(sorted_candidate_tuples_with_distances.keys())[
        :limit
    ]
    for vector_1, vector_2 in relevant_candidate_tuples:
        if vector_1 not in candidates:
            candidates[vector_1] = []
        candidates[vector_1].append(vector_2)
    candidates_end = time.time()
    print(
        f"execution time depth search: {round((candidates_end - candidates_start), 2)}s"
    )
    return candidates


def bitwise_xor_candidates(
    vectors_dict, limit, search_method="depth", print_execution_time=True
):
    start = time.time()
g    packed_vectors = np.packbits(np.array(list(vectors_dict.values())), axis=1)
    if search_method == "breadth":
        candidates = breadth_bitwise_xor_candidates(vectors_dict, limit, packed_vectors)
    else:
        candidates = depth_bitwise_xor_candidates(vectors_dict, limit, packed_vectors)
    end = time.time()
    runtime = round((end - start), 2)
    if print_execution_time:
        print(f"execution time {limit}: {runtime}s")
    return candidates, runtime
