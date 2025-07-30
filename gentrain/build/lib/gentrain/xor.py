import numpy as np
import random
import faiss
import time


def get_lsh_hash(binary_vector, random_indices):
    sampled_bits = [binary_vector[i] for i in random_indices]
    hash_value = "".join(map(str, sampled_bits))
    return hash_value

def add_to_lsh_table(lsh_table, vector_id, vector, random_indices):
    bucket_id = get_lsh_hash(vector, random_indices)
    if bucket_id not in lsh_table:
        lsh_table[bucket_id] = []
    if vector_id not in lsh_table[bucket_id]:
        lsh_table[bucket_id].append(vector_id)
    return lsh_table

def xor_lsh_dynamic(vectors_dict, forced_computation_rate):
    start = time.time()
    vectors_dict = vectors_dict.copy()
    total_distance_count = (len(vectors_dict) * (len(vectors_dict) - 1)) / 2
    first_key = next(iter(vectors_dict))
    vector_length = len(vectors_dict[first_key])
    candidates = {vector_id: [] for vector_id in vectors_dict}
    computation_rate = 0
    iterations = 0
    considered_distance_count = 0
    trim_factor = int(vector_length*0.05)
    hash_length = vector_length
    saturation = 0
    while computation_rate < forced_computation_rate and saturation < 5:
        hash_length = hash_length - trim_factor
        learning_rate = 1
        while computation_rate < forced_computation_rate and learning_rate > 0.01:
            iterations = iterations + 1
            lsh_table = {}
            random_indices = random.sample(range(vector_length), hash_length)
            for vector_id, vector in vectors_dict.items():
                lsh_table = add_to_lsh_table(lsh_table, vector_id, vector, random_indices)
            for _, vector_ids in lsh_table.items():
                for vector_id in vector_ids:
                    candidates_set = set(candidates[vector_id])
                    candidates_set.update(vector_ids)
                    candidates[vector_id] = list(candidates_set)
            updated_considered_distances = set()
            for vector_id, vector_candidates in candidates.items():
                for candidate_id in vector_candidates:
                    updated_considered_distances.add((min(vector_id, candidate_id), max(vector_id, candidate_id)))
            updated_considered_distance_count = len(updated_considered_distances)
            computation_rate = updated_considered_distance_count / total_distance_count
            learning_rate = 1 - (considered_distance_count / updated_considered_distance_count)
            considered_distance_count = updated_considered_distance_count
            print(f"iteration: {iterations}, hash_length: {hash_length}, computation_rate: {computation_rate}, learning_rate: {learning_rate}")
        saturation = saturation + 1 if learning_rate == 0 else 0

    end = time.time()
    runtime = round((end - start), 2)
    print(f"computation rate: {round(computation_rate, 2)}, execution time: {runtime}s, executed iterations: {iterations}")
    print(f"----------------------------------")
    return candidates, runtime

# O(n)
def xor_lsh_hash_length(vectors_dict, hash_length, min_learning_rate):
    start = time.time()
    vectors_dict = vectors_dict.copy()
    total_distance_count = (len(vectors_dict) * (len(vectors_dict) - 1)) / 2
    first_key = next(iter(vectors_dict))
    vector_length = len(vectors_dict[first_key])
    candidates = {vector_id: [] for vector_id in vectors_dict}
    computation_rate = 0
    considered_distance_count = 0
    learning_rate_saturation = 0
    saturation_insurance = 5
    print(f"----------------------------------")
    print(f"hash length: {hash_length}, learning rate theshold: {min_learning_rate}")

    iterations = 0

    while learning_rate_saturation < saturation_insurance and computation_rate < 0.4:
        iterations = iterations + 1
        lsh_table = {}
        random_indices = random.sample(range(vector_length), hash_length)
        for vector_id, vector in vectors_dict.items():
            lsh_table = add_to_lsh_table(lsh_table, vector_id, vector, random_indices)
        for _, vector_ids in lsh_table.items():
            for vector_id in vector_ids:
                candidates_set = set(candidates[vector_id])
                candidates_set.update(vector_ids)
                candidates[vector_id] = list(candidates_set)
        updated_considered_distances = set()
        for vector_id, vector_candidates in candidates.items():
            for candidate_id in vector_candidates:
                updated_considered_distances.add((min(vector_id, candidate_id), max(vector_id, candidate_id)))
        updated_considered_distance_count = len(updated_considered_distances)
        learning_rate = 1 - (considered_distance_count / updated_considered_distance_count)
        considered_distance_count = updated_considered_distance_count
        computation_rate = considered_distance_count / total_distance_count
        if learning_rate < min_learning_rate:
            learning_rate_saturation = learning_rate_saturation + 1
        else:
            learning_rate_saturation = 0

    #min_k_nearest_neighbours = int(len(vectors_dict) * 0.05)
    #fallback_candidates, fallback_runtime = get_k_nearest_neighbors(vectors_dict, min_k_nearest_neighbours)
    #for vector_id in vectors_dict:
    #    vector_candidates = candidates[vector_id]
    #    candidates[vector_id] = list(set(list(vector_candidates) + list(fallback_candidates[vector_id]))) if len(vector_candidates) < min_k_nearest_neighbours else vector_candidates
    #    candidates[vector_id] = [candidate_id for candidate_id in candidates[vector_id] if candidate_id != vector_id]
    end = time.time()
    runtime = round(end - start, 2)
    print(
        f"computation rate: {round(computation_rate, 2)}, execution time: {runtime}s, executed iterations: {iterations}")
    print(f"----------------------------------")
    return candidates, runtime


def xor_lsh_candidates(vectors_dict, hash_lengths, min_learning_rates):
    lsh_candidates = {}
    runtime = {}
    for hash_length in hash_lengths:
        lsh_candidates[hash_length] = {}
        runtime[hash_length] = {}
        for min_learning_rate in min_learning_rates:
            lsh_candidates[hash_length][min_learning_rate], runtime[hash_length]["runtime"] = xor_lsh(vectors_dict,
                                                                                                      hash_length,
                                                                                                      min_learning_rate)
    return lsh_candidates, runtime

def get_k_nearest_neighbors(vectors_dict, k):
    start = time.time()
    first_key = next(iter(vectors_dict))
    vector_length = len(vectors_dict[first_key])
    index = faiss.IndexFlatL2(vector_length)
    index.add(np.array(list(vectors_dict.values()), dtype=np.float32))
    candidates = {}
    for vector_id, vector in vectors_dict.items():
        _, faiss_candidates = index.search(np.array([vector]), k)
        candidates[vector_id] = faiss_candidates[0]
    end = time.time()
    return candidates, round((end - start), 2)

def faiss_cluster_candidates(vectors_dict, limit, cluster_labels, index, candidates):
    candidate_tuples_with_distances = {}
    fallback_tuples_with_distances = {}
    for vector_id, vector in vectors_dict.items():
        faiss_distances, faiss_candidates = index.search(np.array([vector]), len(vectors_dict))
        filtered_distances = []
        filtered_candidates = []
        for candidate_index, candidate in enumerate(faiss_candidates[0]):
            if cluster_labels[vector_id] != -1 and cluster_labels[vector_id] == cluster_labels[candidate]:
                filtered_distances.append(faiss_distances[0][candidate_index])
                filtered_candidates.append(int(candidate))
            else:
               fallback_tuples_with_distances[min(vector_id, candidate), max(vector_id, candidate)] = int(faiss_distances[0][candidate_index])
        for distance_index, distance in enumerate(filtered_distances):
            candidate_id = filtered_candidates[distance_index]
            candidate_tuples_with_distances[min(vector_id, candidate_id), max(vector_id, candidate_id)] = distance
    sorted_candidate_tuples_with_distances = dict(sorted(candidate_tuples_with_distances.items(), key=lambda item: item[1]))
    candidate_tuples = list(sorted_candidate_tuples_with_distances.keys())
    fallback_tuples = list(fallback_tuples_with_distances.keys())
    relevant_candidate_tuples = (candidate_tuples + fallback_tuples)[:limit]
    for vector_1, vector_2 in relevant_candidate_tuples:
        if vector_1 not in candidates:
            candidates[vector_1] = []
        candidates[vector_1].append(vector_2)
    
    return candidates
    
def faiss_depth_search(vectors_dict, limit, index):
    candidates = {}
    distances = []
    candidate_tuples_with_distances = {}
    for vector_id, vector in vectors_dict.items():
        faiss_distances, faiss_candidates = index.search(np.array([vector]), len(vectors_dict))
        vector_distances = {}
        for candidate_index, distance in enumerate(faiss_distances[0]):
            candidate_id = faiss_candidates[0][candidate_index]
            vector_distances[candidate_id] = distance
            if candidate_id == vector_id:
                continue
            candidate_tuples_with_distances[min(vector_id, candidate_id), max(vector_id, candidate_id)] = distance
        distances.append([vector_distances[index] for index in sorted(vector_distances.keys())])

    sorted_candidate_tuples_with_distances = dict(
        sorted(candidate_tuples_with_distances.items(), key=lambda item: item[1]))
    relevant_candidate_tuples = list(sorted_candidate_tuples_with_distances.keys())[:limit]
    for vector_1, vector_2 in relevant_candidate_tuples:
        if vector_1 not in candidates:
            candidates[vector_1] = []
        candidates[vector_1].append(vector_2)
    return candidates, distances
    
def faiss_breadth_search(vectors_dict, limit, index):
    candidates = {}
    distances = []
    candidate_tuples_with_distances = {}
    for vector_id, vector in vectors_dict.items():
        faiss_distances, faiss_candidates = index.search(np.array([vector]), len(vectors_dict))
        vector_distances = {}
        for candidate_index, distance in enumerate(faiss_distances[0]):
            candidate_id = faiss_candidates[0][candidate_index]
            vector_distances[candidate_id] = distance
            if candidate_id == vector_id:
                continue
            candidate_tuples_with_distances[min(vector_id, candidate_id), max(vector_id, candidate_id)] = distance
        distances.append([vector_distances[index] for index in sorted(vector_distances.keys())])

    sorted_candidate_tuples_with_distances = dict(
        sorted(candidate_tuples_with_distances.items(), key=lambda item: item[1]))
    candidate_tuples = list(sorted_candidate_tuples_with_distances.keys())
    distances_count = 0
    
    for vector_1, vector_2 in candidate_tuples:
        if vector_1 not in candidates:
            candidates[vector_1] = []
        if len(candidates[vector_1]) <= int(limit/len(vectors_dict)):
            candidates[vector_1].append(vector_2)                                     
            
    for i in candidate_tuples[distances_count+1]:
        vector_1, vector_2 = candidate_tuples[distances_count+1]
        candidates[vector_1].append(vector_2)
        distances_count += 1
                
    return candidates, distances
    
def k_faiss_exact_cluster_search_candidates(vectors_dict, limit, cluster_labels, centroid_ids, print_execution_time=True):
    start = time.time()
    candidates = {}
    for centroid_id in centroid_ids:
        candidates[centroid_id] = [x for x in centroid_ids if x != centroid_id]
    first_key = next(iter(vectors_dict))
    vector_length = len(vectors_dict[first_key])
    index = faiss.IndexFlatL2(vector_length)
    index.add(np.array(list(vectors_dict.values()), dtype=np.float32))
    candidates = faiss_cluster_candidates(vectors_dict, limit, cluster_labels, index, candidates)
    end = time.time()
    runtime = round((end - start), 2)
    if print_execution_time:
        print(f"execution time {limit}: {runtime}s")
    return candidates, runtime

def faiss_l2_candidates(vectors_dict, limit, search_method="depth", print_execution_time=True):
    start = time.time()
    first_key = next(iter(vectors_dict))
    vector_length = len(vectors_dict[first_key])
    index = faiss.IndexFlatL2(vector_length)
    index.add(np.array(list(vectors_dict.values()), dtype=np.float32))
    if search_method == "breadth":
        candidates = faiss_breadth_search(encodings, limit, index)
    else:
        candidates = faiss_depth_search(encodings, limit, index)
    end = time.time()
    runtime = round((end - start), 2)
    if print_execution_time:
        print(f"execution time {limit}: {runtime}s")
    return candidates, distances, runtime

def faiss_hamming_candidates(vectors_dict, limit, search_method="depth", print_execution_time=True):
    start = time.time()
    first_key = next(iter(vectors_dict))
    vector_length = len(vectors_dict[first_key])
    index = faiss.IndexBinaryFlat(vector_length*8)
    index.add(np.array(list(vectors_dict.values())))
    if search_method == "breadth":
        candidates, distances = faiss_breadth_search(vectors_dict, limit, index)
    else:
        candidates, distances = faiss_depth_search(vectors_dict, limit, index)
    end = time.time()
    runtime = round((end - start), 2)
    if print_execution_time:
        print(f"execution time {limit}: {runtime}s")
    return candidates, distances, runtime