import numpy as np
import random
import faiss
import time
from gentrain.distance_matrix import get_bitwise_xor_distance_matrix
import hnswlib
from collections import defaultdict
from itertools import combinations


def flex_and_or_lsh(encodings, limit):
    start = time.time()
    candidates = {index: [] for index in encodings}
    vector_length = len(encodings[0])
    candidate_tuples = set()
    hash_substractor = int(vector_length * (1/8))
    hash_length = vector_length - hash_substractor
    learning_rate_threshold = 0.1
    learning_rate = 1.0
    while len(candidate_tuples) < limit:
        prev_candidate_tuple_count = len(candidate_tuples)
        lsh_table = {}
        hash_function = random.sample(range(vector_length), hash_length)
        for index, encoding in encodings.items():
            lsh_table = add_to_lsh_table(lsh_table, index, encoding, hash_function)
        for _, indices in lsh_table.items():
            for index_1 in indices:
                for index_2 in indices:
                    if index_1 != index_2:
                        candidate_tuples.add(
                            (min(index_1, index_2), max(index_1, index_2))
                        )
        learning_rate = 1.0 - (prev_candidate_tuple_count / len(candidate_tuples))
        print(hash_length, learning_rate, len(candidate_tuples)/ limit)
        if learning_rate < learning_rate_threshold:
            hash_length = hash_length - hash_substractor

    candidate_tuples = list(candidate_tuples)[:limit]
    for index_1, index_2 in candidate_tuples:
        candidates[index_1].append(index_2)
        candidates[index_2].append(index_1)
    candidates = dict(sorted(candidates.items()))
    end = time.time()
    print(f"xor lsh exectution time: {round((end - start), 2)}")
    return candidates

def get_lsh_hash(binary_vector, random_indices):
    sampled_bits = [binary_vector[i] for i in random_indices]
    hash_value = "".join(map(str, sampled_bits))
    return hash_value


def add_to_lsh_table(lsh_table, vector_id, vector, hash_function):
    bucket_id = get_lsh_hash(vector, hash_function)
    if bucket_id not in lsh_table:
        lsh_table[bucket_id] = []
    if vector_id not in lsh_table[bucket_id]:
        lsh_table[bucket_id].append(vector_id)
    return lsh_table

def and_or_lsh(encodings, hash_length, iterations):
    start = time.time()
    candidates = defaultdict(set)
    vector_length = len(next(iter(encodings.values())))
    for iteration in range(iterations):
        hash_function = random.sample(range(vector_length), hash_length)
        lsh_table = defaultdict(list)

        for index, encoding in encodings.items():
            sampled_bits = [encoding[i] for i in hash_function]
            hash_value = tuple(sampled_bits) 
            lsh_table[hash_value].append(index)

        for indices in lsh_table.values():
            for a, b in combinations(indices, 2):
                candidates[a].add(b)
                candidates[b].add(a)

    candidates = {k: sorted(v) for k, v in sorted(candidates.items())}
    end = time.time()
    print(f"xor lsh execution time: {round((end - start), 2)}")
    return candidates
    

def faiss_k_candidates(vectors_dict, k, index):
    candidates = {}
    for vector_id, vector in vectors_dict.items():
        _, faiss_candidates = index.search(np.array([vector]), k + 1)
        faiss_candidates = faiss_candidates[0]
        faiss_candidates = [
            candidate_id
            for candidate_id in faiss_candidates
            if candidate_id != vector_id
        ]
        candidates[vector_id] = faiss_candidates[:k]
    return candidates


def k_faiss_hnsw_candidates(vectors_dict, k_nearest_neighbors):
    faiss_candidates = {}
    index = faiss.IndexHNSWBinaryB(64, 32)
    index.train(np.array(list(vectors_dict.values()), dtype=np.float32))
    index.add(np.array(list(vectors_dict.values()), dtype=np.float32))
    for k in k_nearest_neighbors:
        faiss_candidates[k] = faiss_k_candidates(vectors_dict, k, index)
    return faiss_candidates


def faiss_cluster_candidates(vectors_dict, limit, cluster_labels, index, candidates):
    candidate_tuples_with_distances = {}
    fallback_tuples_with_distances = {}
    for vector_id, vector in vectors_dict.items():
        faiss_distances, faiss_candidates = index.search(
            np.array([vector]), len(vectors_dict)
        )
        filtered_distances = []
        filtered_candidates = []
        for candidate_index, candidate in enumerate(faiss_candidates[0]):
            if (
                cluster_labels[vector_id] != -1
                and cluster_labels[vector_id] == cluster_labels[candidate]
            ):
                filtered_distances.append(faiss_distances[0][candidate_index])
                filtered_candidates.append(int(candidate))
            else:
                fallback_tuples_with_distances[
                    min(vector_id, candidate), max(vector_id, candidate)
                ] = int(faiss_distances[0][candidate_index])
        for distance_index, distance in enumerate(filtered_distances):
            candidate_id = filtered_candidates[distance_index]
            candidate_tuples_with_distances[
                min(vector_id, candidate_id), max(vector_id, candidate_id)
            ] = distance
    sorted_candidate_tuples_with_distances = dict(
        sorted(candidate_tuples_with_distances.items(), key=lambda item: item[1])
    )
    candidate_tuples = list(sorted_candidate_tuples_with_distances.keys())
    fallback_tuples = list(fallback_tuples_with_distances.keys())
    relevant_candidate_tuples = (candidate_tuples + fallback_tuples)[:limit]
    for vector_1, vector_2 in relevant_candidate_tuples:
        if vector_1 not in candidates:
            candidates[vector_1] = []
        candidates[vector_1].append(vector_2)

    return candidates
    
def get_hnsw_candidates(
    encodings, limit, search_method="depth", print_execution_time=True
):
    start = time.time()
    dim = 128
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
    distances_count = 0

    for vector_1, vector_2 in candidate_tuples:
        if len(candidates[vector_1]) <= int(limit / len(vectors_dict)):
            candidates[vector_1].append(vector_2)

    for i in candidate_tuples[distances_count + 1]:
        vector_1, vector_2 = candidate_tuples[distances_count + 1]
        candidates[vector_1].append(vector_2)
        distances_count += 1

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
    first_key = next(iter(vectors_dict))
    vector_length = len(vectors_dict[first_key])
    packed_vectors = np.packbits(np.array(list(vectors_dict.values())), axis=1)
    if search_method == "breadth":
        candidates = breadth_bitwise_xor_candidates(vectors_dict, limit, packed_vectors)
    else:
        candidates = depth_bitwise_xor_candidates(vectors_dict, limit, packed_vectors)
    end = time.time()
    runtime = round((end - start), 2)
    if print_execution_time:
        print(f"execution time {limit}: {runtime}s")
    return candidates, runtime
