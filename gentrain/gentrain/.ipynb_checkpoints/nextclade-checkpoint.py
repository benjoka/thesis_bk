import pkgutil
from io import StringIO

import pandas as pd
from Bio import SeqIO

reference_fasta_bytes = pkgutil.get_data(__package__, "reference.fasta")
reference_fasta_io = StringIO(reference_fasta_bytes.decode())
for record in SeqIO.parse(reference_fasta_io, "fasta"):
    reference_sequence = str(record.seq)

def get_mutations_from_dataframe(dataframe):
    mutations = dataframe[
        ["substitutions", "insertions", "deletions", "missing", "nonACGTNs"]]
    mutations = mutations.apply(lambda col: col.map(lambda x: x.split(",") if isinstance(x, str) else []))
    return mutations

def get_mutations_from_csv(csv_path):
    dataframe = pd.read_csv(csv_path, delimiter=";", low_memory=False)
    return get_mutations_from_dataframe(dataframe)
