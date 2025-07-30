import numpy as np
from scipy.stats import kendalltau
import time

def get_bitwise_xor_distance_matrix(encodings):
    start = time.time()
    hamming_distance_matrix = []
    packed_vectors = np.packbits(np.array(list(encodings.values())), axis=1)
    for isolate_id, isolate_encoding in encodings.items():
        xor_result = np.bitwise_xor(packed_vectors, packed_vectors[isolate_id])
        hamming_distance_matrix.append(np.unpackbits(xor_result, axis=1).sum(axis=1))
    hamming_distance_matrix = np.array(hamming_distance_matrix)
    end = time.time()
    print(f"matrix generation time: {round(end - start, 2)}s")
    return hamming_distance_matrix

def get_kendall_tau_correlation(matrix_1, matrix_2):
    triu_mask = np.triu(np.ones_like(matrix_1, dtype=bool), k=1)
    matrix_1 = matrix_1[triu_mask]
    matrix_2 = matrix_2[triu_mask]
    correlation, p = kendalltau(matrix_1, matrix_2)
    return correlation

def get_signed_rmse(matrix_1, matrix_2):
    triu_mask = np.triu(np.ones_like(matrix_1, dtype=bool), k=1)
    matrix_1 = matrix_1[triu_mask]
    matrix_2 = matrix_2[triu_mask]
    mean_error = np.mean(matrix_2 - matrix_1)
    rmse = np.sqrt(np.mean((matrix_2 - matrix_1) ** 2))
    return np.sign(mean_error) * rmse
    
def get_signed_infection_rmse(matrix_1, matrix_2):
    triu_mask = np.triu(np.ones_like(matrix_1, dtype=bool), k=1)
    matrix_1 = matrix_1[triu_mask]
    matrix_2 = matrix_2[triu_mask]
    infection_mask_1 = matrix_1 < 2
    matrix_1 = matrix_1[infection_mask_1]
    matrix_2 = matrix_2[infection_mask_1]
    mean_error = np.mean(matrix_2 - matrix_1)
    rmse = np.sqrt(np.mean((matrix_2 - matrix_1) ** 2))
    return np.sign(mean_error) * rmse
    
def get_infection_recall(matrix_1, matrix_2):
    triu_mask = np.triu(np.ones_like(matrix_1, dtype=bool), k=1)
    matrix_1 = matrix_1[triu_mask]
    matrix_2 = matrix_2[triu_mask]
    infection_mask_1 = matrix_1 < 2
    infection_mask_2 = matrix_2 < 2
    not_infection_mask_2 = matrix_2 >= 2
    tp = np.sum(infection_mask_1 & infection_mask_2)
    fn = np.sum(infection_mask_1 & not_infection_mask_2)

    return tp/(tp+fn)

def get_infection_precision(matrix_1, matrix_2):
    triu_mask = np.triu(np.ones_like(matrix_1, dtype=bool), k=1)
    matrix_1 = matrix_1[triu_mask]
    matrix_2 = matrix_2[triu_mask]
    not_infection_mask_1 = matrix_1 >= 2
    infection_mask_1 = matrix_1 < 2
    infection_mask_2 = matrix_2 < 2
    tp = np.sum(infection_mask_1 & infection_mask_2)
    fp = np.sum(not_infection_mask_1 & infection_mask_2)
    return tp/(tp+fp)
    
def get_infection_f1(matrix_1, matrix_2):
    triu_mask = np.triu(np.ones_like(matrix_1, dtype=bool), k=1)
    matrix_1 = matrix_1[triu_mask]
    matrix_2 = matrix_2[triu_mask]
    not_infection_mask_1 = matrix_1 >= 2
    infection_mask_1 = matrix_1 < 2
    not_infection_mask_2 = matrix_2 >= 2
    infection_mask_2 = matrix_2 < 2
    tp = np.sum(infection_mask_1 & infection_mask_2)
    tn = np.sum(not_infection_mask_1 & not_infection_mask_2)
    fp = np.sum(not_infection_mask_1 & infection_mask_2)
    fn = np.sum(infection_mask_1 & not_infection_mask_2)
    return (2*tp) /((2*tp)+fp+fn)

def median_distance(matrix):
    triu_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    matrix = matrix[triu_mask]
    return np.median(matrix)



