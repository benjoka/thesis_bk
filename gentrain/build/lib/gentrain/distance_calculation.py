from Bio import Align

import numpy as np
from Bio import SeqIO

ambiguous_chars = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "U": ["U"],
    "M": ["A", "C"],
    "R": ["A", "G"],
    "S": ["C", "G"],
    "W": ["A", "T"],
    "Y": ["C", "T"],
    "K": ["G", "T"],
    "V": ["A", "C", "G"],
    "H": ["A", "C", "T"],
    "D": ["A", "G", "T"],
    "B": ["C", "G", "T"],
    "N": ["A", "C", "G", "T"],
    "X": ["A", "C", "G", "T"],
}

reference_sequence = ""
for record in SeqIO.parse(("../gentrain/gentrain/reference.fasta"), "fasta"):
    reference_sequence = str(record.seq)
    reference_sequence_array = np.array(list(reference_sequence), dtype=str)

aligner = Align.PairwiseAligner(match_score=1.0)


def prepend_manual(values, arr):
    values = np.atleast_1d(values)
    new_arr = np.empty(len(arr) + len(values), dtype=arr.dtype)
    new_arr[:len(values)] = values
    new_arr[len(values):] = arr
    return new_arr

def get_mutation_positions(sequence_mutations):
    mutation_positions = {}
    for substitution in sequence_mutations["substitutions"]:
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', substitution)
        position = int(match.group(2))
        character = match.group(3)
        if position not in mutation_positions:
            mutation_positions[position] = {}
        mutation_positions[position]["snp"] = character
    for insertion in sequence_mutations["insertions"]:
        match = re.match(r'^(\d+):([A-Z]+)$', insertion)
        position = int(match.group(1))
        characters = list(match.group(2))
        if position not in mutation_positions:
            mutation_positions[position] = {}
        mutation_positions[position]["ins"] = ''.join(characters)
    for deletion in sequence_mutations["deletions"]:
        match = re.match(r'^(\d+)(?:-(\d+))?$', deletion)
        deletion_start = int(match.group(1))
        deletion_end = int(match.group(2)) if match.group(2) else int(match.group(1))
        for position in range(deletion_start, deletion_end + 1):
            if position not in mutation_positions:
                mutation_positions[position] = {}
            mutation_positions[position]["del"] = "-"
    return mutation_positions
    
def align_samples(mutation_positions_1, mutation_positions_2):
    sequence_1 = np.array([], dtype=str)
    sequence_2 = np.array([], dtype=str)

    for current_base_index in range(len(reference_sequence_array) - 1):
        reference_index_char = reference_sequence_array[current_base_index]
        additions_1 = np.array([], dtype=str)
        additions_2 = np.array([], dtype=str)

        # add remaining reference characters if index is not in positions dictionary
        if current_base_index not in mutation_positions_1:
            additions_1 = np.append(additions_1, reference_index_char)

        if current_base_index not in mutation_positions_2:
            additions_2 = np.append(additions_2, reference_index_char)

        # get mutations for current base index or return empty list if no mutations for current index exist
        position_mutations1 = mutation_positions_1[
            current_base_index] if current_base_index in mutation_positions_1 else {}
        position_mutations2 = mutation_positions_2[
            current_base_index] if current_base_index in mutation_positions_2 else {}

        # add snp characters for both sequences
        if "snp" in position_mutations1:
            additions_1 = np.append(additions_1, position_mutations1["snp"])
        if "snp" in position_mutations2:
            additions_2 = np.append(additions_2, position_mutations2["snp"])

        # handle deletions for current base index, respecting the others sequence mutations
        if not ("del" in position_mutations1 and "del" in position_mutations2) and not (
                "del" not in position_mutations1 and "del" not in position_mutations2):
            if "del" in position_mutations1:
                additions_1 = np.append(additions_1, position_mutations1["del"])
            if "del" in position_mutations2:
                additions_2 = np.append(additions_2, position_mutations2["del"])

        # handle insertions
        if "ins" in position_mutations1 and "ins" in position_mutations2:
            if position_mutations1["ins"] != position_mutations2["ins"]:
                # alignments = aligner.align(position_mutations1["ins"], position_mutations2["ins"])
                #additions_1 = np.append(additions_1, position_mutations1["ins"])
                #additions_2 = np.append(additions_2, position_mutations2["ins"])
                additions_1 = additions_1
                additions_2 = additions_2
            else:
                if len(position_mutations1) > 1:
                    additions_1 = np.append(additions_1, position_mutations1["ins"])
                else:
                    additions_1 = np.append(additions_1, position_mutations1["ins"])
                    additions_1 = prepend_manual(reference_index_char, additions_1)
                if len(position_mutations2) > 1:
                    additions_2 = np.append(additions_2, position_mutations2["ins"])
                else:
                    additions_2 = np.append(additions_2, position_mutations2["ins"])
                    additions_2 = prepend_manual(reference_index_char, additions_2)
        elif "ins" in position_mutations1:
            if len(position_mutations1) > 1:
                additions_1 = np.append(additions_1, position_mutations1["ins"])
            else:
                additions_1 = np.append(additions_1, position_mutations1["ins"])
                additions_1 = prepend_manual(reference_index_char, additions_1)
            additions_2 = np.append(additions_2, ["-" for x in range(len(position_mutations1["ins"]))])
        elif "ins" in position_mutations2:
            if len(position_mutations2) > 1:
                additions_2 = np.append(additions_2, position_mutations2["ins"])
            else:
                additions_2 = np.append(additions_2, position_mutations2["ins"])
                additions_2 = prepend_manual(reference_index_char, additions_2)
            additions_1 = np.append(additions_1, ["-" for x in range(len(position_mutations2["ins"]))])

        # add new additions to prior sequences
        sequence_1 = np.append(sequence_1, additions_1)
        sequence_2 = np.append(sequence_2, additions_2)
    return sequence_1, sequence_2


def calculate_distance(sequence_1, sequence_2):
    distance = 0
    proper_threshold = 5
    proper_chars_1 = 0
    proper_chars_2 = 0
    n_count_1 = np.count_nonzero(sequence_1 == "N")
    n_count_2 = np.count_nonzero(sequence_2 == "N")
    sequence_length_1 = sequence_1.size
    sequence_length_2 = sequence_2.size
    active_gap_1 = False
    active_gap_2 = False
    for current_base_index in range(sequence_length_1 - 1):
        current_char_1 = sequence_1[current_base_index]
        current_char_2 = sequence_2[current_base_index]
        if current_char_1 != "-":
            active_gap_1 = False
        if current_char_2 != "-":
            active_gap_2 = False

        # dont increment distance, dont increment proper_chars
        if current_char_1 == "N" or current_char_2 == "N":
            continue

        # increment proper_chars if current char is not "-"
        proper_chars_1 = proper_chars_1 + 1 if current_char_1 != "-" else proper_chars_1 + 0
        proper_chars_2 = proper_chars_2 + 1 if current_char_2 != "-" else proper_chars_2 + 0

        # dont increment distance
        if current_char_1 == current_char_2:
            continue

        # continue if amount of proper chars is not yet reached to increment distances
        if (proper_chars_1 < proper_threshold) or (
                (sequence_length_1 - n_count_1) - proper_chars_1 < proper_threshold) or (
                proper_chars_2 < proper_threshold) or (
                (sequence_length_2 - n_count_2) - proper_chars_2 < proper_threshold):
            continue

        # increment distance on gap
        if current_char_1 == "-":
            if not active_gap_1:
                distance = distance + 1
        if current_char_2 == "-":
            if not active_gap_2:
                distance = distance + 1

        # increment distance if not gap and differing chars
        if current_char_1 != "-" and current_char_2 != "-" and not (
                current_char_1 in ambiguous_characters[current_char_2] or current_char_2 in ambiguous_characters[
            current_char_1]):
            distance = distance + 1

    return distance


def get_distance_for_two_samples(mutation_positions_1, mutation_positions_2):
    sequence_1, sequence_2 = align_samples(mutation_positions_1, mutation_positions_2)
    return calculate_distance(sequence_1, sequence_2)
