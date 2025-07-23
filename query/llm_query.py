"""Structured and semantic search for LLM data."""

import numpy as np
import polars as pl
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from analysis import compute_ranking_for_subset
from analysis import (
    load_instances,
    load_pairwise_results,
    load_embeddings,
)
from query import embed
import faiss  # needs to be imported after sentence_transformers


@hydra.main(config_path="configs", config_name="main", version_base="1.1")
def query_and_rank(cfg: DictConfig):

    pairwise_results = load_pairwise_results()
    _, uuids = query(cfg)
    ranking = compute_ranking_for_subset(pairwise_results, uuids)

    return ranking


def query(cfg: DictConfig):
    instances_dict = load_instances()
    instances_list = [{"id": k, **v} for k, v in instances_dict.items()]
    instances_df = pl.DataFrame(pd.json_normalize(instances_list))

    filters = parse_filters(cfg.filters)
    instances_df, indices = apply_filters(instances_df, filters)
    if indices.size == 0:
        print("No instances found.")
        return None, None
    uuids = instances_df["id"].to_list()

    if cfg.query is not None:
        embeddings = load_embeddings(benchmark="all")[indices]
        indices, similarities = search(cfg, embeddings)
        question_uuids = instances_df[indices, "id"].to_list()
        question_similarities = similarities

    if cfg.answer_query is not None:
        answer_embeddings = load_embeddings(benchmark="all", type="answer")[indices]
        indices, similarities = search(cfg, answer_embeddings)
        answer_uuids = instances_df[indices, "id"].to_list()
        answer_similarities = similarities

    if cfg.query is not None and cfg.answer_query is not None:
        final_uuids = question_uuids.intersection(answer_uuids)
        final_similarities = np.array(
            [question_similarities[question_uuids.index(uuid)] for uuid in final_uuids]
        )
    elif cfg.query is not None:
        final_uuids = question_uuids
        final_similarities = question_similarities
    elif cfg.answer_query is not None:
        final_uuids = answer_uuids
        final_similarities = answer_similarities
    else:
        final_uuids = set(uuids)
        final_similarities = np.ones(len(final_uuids))

    if not final_uuids:
        print("No instances match the search criteria.")
        return None, None

    return final_similarities, list(final_uuids)


def search(cfg, embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if cfg.top_k is not None and cfg.threshold is not None:
        similarities, indices = index.search(embed([cfg.query]), cfg.top_k)
        similarities = similarities[0]
        indices = indices[0]
        mask = similarities > cfg.threshold
        similarities = similarities[mask]
        indices = indices[mask]
    elif cfg.top_k is not None:
        similarities, indices = index.search(embed([cfg.query]), cfg.top_k)
        similarities = similarities[0]
        indices = indices[0]
    elif cfg.threshold is not None:
        similarities, indices = index.range_search(
            embed([cfg.query]), radius=cfg.threshold
        )
    else:
        raise ValueError("Either top_k or threshold must be specified.")

    indices = indices[indices != -1]
    return indices, similarities


def parse_filters(filters):
    if filters is None or not filters:
        return None
    op_map = {
        "eq": lambda df, col, val: df[col] == val,
        "ne": lambda df, col, val: df[col] != val,
        "gt": lambda df, col, val: df[col] > val,
        "ge": lambda df, col, val: df[col] >= val,
        "lt": lambda df, col, val: df[col] < val,
        "le": lambda df, col, val: df[col] <= val,
        "contains": lambda df, col, val: df[col].str.contains(val),
        "len": lambda df, col, val: df[col].list.lengths() == val,
    }

    filters = OmegaConf.to_container(filters, resolve=True)
    filters = pd.json_normalize(filters).to_dict(orient="records")[0]
    filters_parsed = []
    for k, value in filters.items():
        *column, op_name = k.split(".")
        column = ".".join(column)
        try:
            op = op_map[op_name]
        except KeyError as e:
            raise ValueError(
                f"Unsupported op: {op_name}. Available: {op_map.keys()}"
            ) from e
        if isinstance(value, list):
            for v in value:
                filters_parsed.append((op, column, v))
        else:
            filters_parsed.append((op, column, value))

    return filters_parsed


def apply_filters(df, filters):
    """Apply filters to the dataframe and return the filtered dataframe and indices."""
    if filters is None:
        return df, np.arange(len(df))

    mask = pl.Series(np.ones(len(df), dtype=bool))
    for function, column, value in filters:
        filter_mask = function(df, column, value)
        mask &= filter_mask

    filtered_df = df.filter(mask)
    indices = mask.to_numpy().nonzero()[0]
    return filtered_df, indices


if __name__ == "__main__":
    query_and_rank()
