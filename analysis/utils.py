import polars as pl
import os
from typing import List
import numpy as np
import pandas as pd

def convert_cols(df: pl.DataFrame) -> pl.DataFrame:
    columns_to_convert = []

    for col in df.columns:
        series = df[col]
        unique_values = series.unique().to_list()

        if all(value in {0.0, 1.0, None} for value in unique_values):
            columns_to_convert.append(series.cast(pl.Int64).alias(col))

    if columns_to_convert:
        df = df.with_columns(columns_to_convert)

    return df

def generate_top1_list(results):
    model_to_index = {model: idx for idx, model in enumerate(results.index)}
    models = results.index
    comparison_outcomes = []

    for column in results.columns:
        series = results[column]
        series = series.dropna()

        unique_values = series.unique()

        if set(unique_values).issubset({0.0, 1.0}) or set(unique_values).issubset({True, False}):
            winners = series[series == 1].index.tolist()

            for winner in winners:
                losers = series[series == 0].index.tolist()
                if len(losers) == 0:
                    continue
                winner_idx = model_to_index[winner]
                loser_idxs = [model_to_index[loser] for loser in losers]
                comparison_outcomes.append([winner_idx, loser_idxs])
        else:
            sorted_series = series.sort_values(ascending=False)
            while not sorted_series.empty:
                max_value = sorted_series.max()
                # winners = sorted_series.filter(sorted_series == max_value).index().to_list()

                winners = sorted_series[sorted_series == max_value].index.to_list()

                sorted_series = sorted_series[sorted_series != max_value]
                sorted_series = sorted_series.filter(sorted_series != max_value)
                losers = sorted_series.index.to_list()

                if len(losers) == 0:
                    continue

                for winner in winners:
                    winner_idx = model_to_index[models[winner]]
                    loser_idxs = [model_to_index[models[loser]] for loser in losers]

                    comparison_outcomes.append([winner_idx, loser_idxs])
    return comparison_outcomes