import polars as pl
from tqdm import tqdm

def load_long(df):
    return df.melt(
        id_vars=["model"], variable_name="data_instance", value_name="score"
    )

def compute_pairwise(pairwise_results):
    """Rank the models using the Plackett-Luce model based on pairwise results."""
    rankings = []

    models = pl.concat([pairwise_results["model_a"], pairwise_results["model_b"]]).unique().to_list()
    model_to_index = {model: idx for idx, model in enumerate(models)}
    for model_a, model_b, winner in pairwise_results[
        ["model_a", "model_b", "winner"]
    ].iter_rows():
        if winner == "model_a":
            rankings.append([model_to_index[model_a], model_to_index[model_b]])
        elif winner == "model_b":
            rankings.append([model_to_index[model_b], model_to_index[model_a]])
        elif winner == "tie":
            rankings.append([model_to_index[model_b], model_to_index[model_a]])
            rankings.append([model_to_index[model_a], model_to_index[model_b]])
    return rankings, models

def to_pairwise(results, long=False):
    """Convert cardinal data in long format to pairwise data."""
    if not long:
        results = results.melt(
            id_vars=["model"], variable_name="data_instance", value_name="score"
        )

    data_instances = results["data_instance"].unique().to_list()

    pairwise = []
    for data_instance in tqdm(data_instances):
        instance_df = results.filter(pl.col("data_instance") == data_instance)

        # Create a self join to get pairs of models for the same data instance
        instance_df = instance_df.join(instance_df, on="data_instance", suffix="_other")

        # Filter out self-pairs and duplicate pairs
        filtered_df = instance_df.filter(pl.col("model") < pl.col("model_other"))

        winner_df = filtered_df.with_columns(
            pl.when(pl.col("score") > pl.col("score_other"))
            .then(pl.lit("model_a"))
            .when(pl.col("score") < pl.col("score_other"))
            .then(pl.lit("model_b"))
            .otherwise(pl.lit("tie"))
            .alias("winner")
        ).select(["model", "model_other", "data_instance", "winner"])
        pairwise.append(winner_df)
    pairwise = pl.concat(pairwise).rename(
        {"model": "model_a", "model_other": "model_b"}
    )
    return pairwise

