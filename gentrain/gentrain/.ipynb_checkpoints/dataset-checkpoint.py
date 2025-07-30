import pandas as pd
import os

def get_data_aggregate_sample(dataframe, size, seasonal_column, aggregate):
    aggregate_path = f"aggregates/{aggregate}"
    if not os.path.exists(aggregate_path):
        os.mkdir(aggregate_path)
    if not os.path.exists(f"{aggregate_path}/{size}"):
        os.mkdir(f"{aggregate_path}/{size}")
    if os.path.exists(f"{aggregate_path}/sequences_and_metadata.csv"):
        shutil.rmtree(f"{aggregate_path}/sequences_and_metadata.csv")
    n_per_interval = int(size/len(dataframe[seasonal_column].value_counts()))
    sample = dataframe.groupby(seasonal_column).apply(lambda x: x.sample(n=min(n_per_interval, len(x)), random_state=42), include_groups=False)
    sample_diff = size - len(sample)
    print("Sample diff: ", sample_diff)
    sample = pd.concat([sample, dataframe[~dataframe["igs_id"].isin(list(sample["igs_id"]))].sample(sample_diff)], ignore_index=True)
    print("Sample length: ", len(sample))
    full_city_dist = dataframe["prime_diagnostic_lab.city"].value_counts(normalize=True).sort_index()
    sampled_city_dist = sample["prime_diagnostic_lab.city"].value_counts(normalize=True).sort_index()
    distribution_diff = (full_city_dist - sampled_city_dist).abs()
    print("Geographical distribution: ", distribution_diff.mean())
    sample.to_csv(f"aggregates/{aggregate}/{size}/sequences_and_metadata.csv", sep=";")
    return sample