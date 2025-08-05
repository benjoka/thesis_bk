import pandas as pd
import numpy as np
import re
from Bio import SeqIO
from gentrain.nextclade import get_mutations_from_dataframe
from sklearn.metrics.pairwise import cosine_similarity
import pkgutil
from io import StringIO
from sklearn.preprocessing import MultiLabelBinarizer
import itertools
import re
from collections import Counter
import time

reference_sequence = ""
reference_fasta_bytes = pkgutil.get_data(__package__, "reference.fasta")
reference_fasta_io = StringIO(reference_fasta_bytes.decode())
for record in SeqIO.parse(reference_fasta_io, "fasta"):
    reference_sequence = str(record.seq)
    reference_sequence_array = np.array(list(reference_sequence), dtype=str)
    
def get_missing_positions(mutations):
    missing_positions = []
    for missing in mutations["missing"]:
        match = re.match(r"^(\d+)(?:-(\d+))?$", missing)
        start = int(match.group(1))
        end = int(match.group(2)) + 1 if match.group(2) else start + 1
        for position in range(start,end):  
            if position >= mutations["alignmentStart"] or position <= mutations["alignmentEnd"]:
                missing_positions.append(position)
    return missing_positions
    
def filter_mutations_by_missings(mutations, isolates_df, shifted_relevant_mutations, shifted_reference_positions):
    mutations["alignmentStart"] = isolates_df["alignmentStart"]
    mutations["alignmentEnd"] = isolates_df["alignmentEnd"]
    missing_positions = mutations.apply(lambda x: get_missing_positions(x), axis=1)
    all_missing_positions = list(itertools.chain.from_iterable(missing_positions))
    missing_position_appearances = Counter(all_missing_positions)
    filtered_positions = []
    for position in range(len(reference_sequence_array)):
        if missing_position_appearances[position] > (0.05*len(isolates_df)):
            filtered_positions.append(position)
    shifted_filtered_positions = [shifted_reference_positions[position] for position in filtered_positions]
    filtered_relevant_mutations = shifted_relevant_mutations.drop(columns=shifted_filtered_positions, errors="ignore")
    return filtered_relevant_mutations
    
def find_and_shift_relevant_mutation_positions(mutations, isolates_df, minified_aligned_sequences, shifted_reference_positions, exclude_indels, use_frequency_filtering):
    # filter substitions by relevance based on observed behaviours in MST constellations
    substitution_counts = mutations["substitutions"].explode().value_counts().reset_index()
    if use_frequency_filtering:
        substitution_counts = substitution_counts[(substitution_counts["count"]/len(isolates_df) < 0.2) & (substitution_counts["count"] > 1)]
    relevant_substitution_positions = list(substitution_counts["substitutions"].apply(lambda x: int(re.match(r"^([A-Z])(\d+)([A-Z])$", x).group(2))))
    shifted_relevant_substitution_positions = [shifted_reference_positions[position] for position in relevant_substitution_positions]
    if exclude_indels:
        relevant_substitution_positions.sort()
        relevant_shifted_substitutions = minified_aligned_sequences[shifted_relevant_substitution_positions]
        return relevant_shifted_substitutions, shifted_reference_positions
        
    insertion_counts = mutations["insertions"].explode().value_counts().reset_index()
    insertion_positions = list(insertion_counts["insertions"].apply(lambda x: int(re.match(r"^(\d+):([A-Z]+)$", x).group(1))))
    shifted_insertion_positions = [shifted_reference_positions[position] + 1 for position in insertion_positions if position != 0]
    deletion_counts = mutations["deletions"].explode().value_counts().reset_index()
    deletion_positions = list(deletion_counts["deletions"].apply(lambda x: int(re.match(r"^(\d+)(?:-(\d+))?$", x).group(1))))
    shifted_deletion_positions = [shifted_reference_positions[position] for position in deletion_positions]

    relevant_mutation_positions = relevant_substitution_positions + insertion_positions + deletion_positions
    relevant_mutation_positions.sort()
    relevant_shifted_mutations = pd.concat([minified_aligned_sequences[shifted_relevant_substitution_positions], minified_aligned_sequences[shifted_insertion_positions], minified_aligned_sequences[shifted_deletion_positions]], axis=1)
    return relevant_shifted_mutations, shifted_reference_positions

def get_inserted_lengths(unique_insertion_list):
    inserted_lengths = {}
    for insertion in unique_insertion_list:
        regex_result = re.findall(r"[0-9]+|:|[A-Za-z]+", insertion)
        position = int(regex_result[0])
        inserted_chars = regex_result[2]
        if position not in inserted_lengths or position not in inserted_lengths and len(inserted_lengths[position]) < len(inserted_chars):
            inserted_lengths[position] = 1
    return dict(sorted(inserted_lengths.items()))

def get_shifted_reference_positions(unique_insertion_list):
    shifted_reference_positions = {x:x for x in range(1,len(reference_sequence_array)+1)}
    current_shift = 0
    inserted_lengths = get_inserted_lengths(unique_insertion_list)
    for position in shifted_reference_positions:
        shifted_reference_positions[position] = shifted_reference_positions[position] + current_shift
        # increment shift for the next position, because 3073:GAA inserts GAA between 3073 and 3074 so that 3074 turns into 3077
        if position in inserted_lengths.keys():
            current_shift += inserted_lengths[position]
    return shifted_reference_positions

def get_aligned_reference(unique_insertion_list, shifted_reference_positions):
    aligned_reference = {}
    aligned_sequences_length = list(shifted_reference_positions.values())[-1]+1
    for aligned_position in range(1, aligned_sequences_length+1):
        if aligned_position in list(shifted_reference_positions.values()):
            reference_position = list(shifted_reference_positions.keys())[list(shifted_reference_positions.values()).index(aligned_position)]
            aligned_reference[aligned_position] = reference_sequence[reference_position - 1]
        else:
            aligned_reference[aligned_position] = "-"
    return aligned_reference

def get_aligned_mutation_dict(row, aligned_reference, shifted_reference_positions, exclude_indels=False):
    aligned_sequence = {i: 0 for i in range(1, len(aligned_reference)+1)}
    for substitution in row["substitutions"]:
        match = re.match(r"^([A-Z])(\d+)([A-Z])$", substitution)
        position = int(match.group(2))
        aligned_sequence[shifted_reference_positions[position]] = 1
    if exclude_indels:
        return aligned_sequence
    for insertion in row["insertions"]:
        match = re.match(r"^(\d+):([A-Z]+)$", insertion)
        position = int(match.group(1))
        if position != 0:
            aligned_sequence[shifted_reference_positions[position] + 1] = 1
    for deletion in row["deletions"]:
        match = re.match(r"^(\d+)(?:-(\d+))?$", deletion)
        deletion_start = int(match.group(1))
        aligned_sequence[shifted_reference_positions[deletion_start]] = 1
    return aligned_sequence
    
def get_aligned_nucleotide_dict(row, aligned_reference, shifted_reference_positions, exclude_indels=False):
    aligned_sequence = aligned_reference.copy()
    for substitution in row["substitutions"]:
        match = re.match(r"^([A-Z])(\d+)([A-Z])$", substitution)
        position = int(match.group(2))
        character = match.group(3)
        aligned_sequence[shifted_reference_positions[position]] = character
    if exclude_indels:
        return aligned_sequence
    for insertion in row["insertions"]:
        match = re.match(r"^(\d+):([A-Z]+)$", insertion)
        position = int(match.group(1))
        characters = list(match.group(2))
        if position != 0:
            aligned_sequence[shifted_reference_positions[position] + 1] = characters[0]
    for deletion in row["deletions"]:
        match = re.match(r"^(\d+)(?:-(\d+))?$", deletion)
        deletion_start = int(match.group(1))
        aligned_sequence[shifted_reference_positions[deletion_start]] = "-"
    return aligned_sequence

def get_shifted_reference_positions_from_csv(isolates_df):
    mutations = get_mutations_from_dataframe(isolates_df)
    unique_insertion_list = list(set(mutations["insertions"].explode().dropna().tolist()))
    shifted_reference_positions = get_shifted_reference_positions(unique_insertion_list)
    return shifted_reference_positions
    
def filter_and_align_mutations(isolates_df, mutation_sensitive, exclude_indels, use_frequency_filtering, filter_N):
    mutations = get_mutations_from_dataframe(isolates_df)
    unique_insertion_list = list(set(mutations["insertions"].explode().dropna().tolist()))
    shifted_reference_positions = get_shifted_reference_positions(unique_insertion_list)
    aligned_reference = get_aligned_reference(unique_insertion_list, shifted_reference_positions)
    aligned_sequences_dict = list(mutations.apply(lambda row: get_aligned_mutation_dict(row, aligned_reference, shifted_reference_positions, exclude_indels) if mutation_sensitive else get_aligned_nucleotide_dict(row, aligned_reference, shifted_reference_positions, exclude_indels), axis=1))
    aligned_sequences_df = pd.DataFrame(aligned_sequences_dict)
    minified_aligned_sequences = aligned_sequences_df
    shifted_relevant_mutations, shifted_reference_positions = find_and_shift_relevant_mutation_positions(mutations, isolates_df, minified_aligned_sequences, shifted_reference_positions, exclude_indels, use_frequency_filtering)
    if filter_N:
        shifted_relevant_mutations = filter_mutations_by_missings(mutations, isolates_df, shifted_relevant_mutations, shifted_reference_positions)
    filtered_and_aligned_mutations = shifted_relevant_mutations.loc[:,~shifted_relevant_mutations.columns.duplicated()]
    return filtered_and_aligned_mutations

def trim_vocab_for_faiss(mutations, m):
    columns_to_drop = len(mutations.columns) % m
    cols = []
    if columns_to_drop > 0:
        for col in mutations.columns:
            value_counts = mutations[col].value_counts(dropna=False)
            if len(value_counts) == 2 and (1 in value_counts.values or 2 in value_counts.values):
                cols.append(col)
                columns_to_drop = columns_to_drop - 1
            if columns_to_drop == 0:
                break
    mutations = mutations.drop(cols, axis=1)
    return mutations

def get_mutation_sensitive_encodings(isolates_df, exclude_indels = False, use_frequency_filtering = True, filter_N = True):
    start = time.time()
    filtered_and_aligned_mutations = filter_and_align_mutations(isolates_df, True, exclude_indels, use_frequency_filtering, filter_N)
    mutations = trim_vocab_for_faiss(filtered_and_aligned_mutations, 8)
    char_array = mutations.to_numpy()
    unique_chars = np.unique(char_array)
    char_to_int = {char: idx for idx, char in enumerate(sorted(unique_chars))}
    encode = np.vectorize(lambda x: float(char_to_int[x]))
    encodings = encode(char_array)
    end = time.time()
    print(f"execution time: {round(end - start, 2)}s")
    return {index: np.array(encoding, dtype=np.uint8) for index, encoding in enumerate(encodings)}
    
def get_nucleotide_sensitive_encodings(isolates_df, exclude_indels = False, use_frequency_filtering = True, filter_N = True):
    start = time.time()
    filtered_and_aligned_mutations = filter_and_align_mutations(isolates_df, False, exclude_indels, use_frequency_filtering, filter_N)
    mutations = trim_vocab_for_faiss(filtered_and_aligned_mutations, 8)
    char_array = mutations.to_numpy()
    unique_chars = np.unique(char_array)
    char_to_int = {char: idx for idx, char in enumerate(sorted(unique_chars))}
    encode = np.vectorize(lambda x: float(char_to_int[x]))
    encodings = encode(char_array)
    one_hot_encodings = generate_one_hot_encoding(encodings)    
    end = time.time()
    print(f"execution time: {round(end - start, 2)}s")
    return {index: np.array(encoding, dtype=np.uint8) for index, encoding in enumerate(one_hot_encodings)}

def generate_one_hot_encoding(encodings):
    unique_values = sorted(set(val for encoding in encodings for val in encoding))
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    
    def one_hot_encode(encoding, value_to_index):
        num_classes = len(value_to_index)
        return np.eye(num_classes, dtype="float32")[[value_to_index[val] for val in encoding]]
    
    return [one_hot_encode(encoding, value_to_index).flatten() for encoding in encodings]