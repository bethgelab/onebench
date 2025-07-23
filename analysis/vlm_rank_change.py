import pandas as pd
import polars as pl
import os
from elo import compute_mle, to_pairwise

VHELM_DIR = "data/vlm/vhelm/"
ARENA_DIR = "data/vlm/"


def load_vhelm_results_long():
    """Load the LLM evaluation results in the long format."""
    results_path1 = VHELM_DIR + "binary.parquet"
    bin = pl.read_parquet(results_path1).melt(
        id_vars=["model"], variable_name="data_instance", value_name="score"
    )
    print(bin.shape)
    results_path2 = VHELM_DIR + "numeric.parquet"
    num = pl.read_parquet(results_path2).melt(
        id_vars=["model"], variable_name="data_instance", value_name="score"
    )
    print(num.shape)
    results = pl.concat([bin, num])
    print(results.shape)
    return results


def load_vhelm_results_pairwise():
    """Load or compute the pairwise model comparisons."""
    pairwise_results_path = VHELM_DIR + "pairwise.parquet"
    # Check if path exists
    if os.path.exists(pairwise_results_path):
        pairwise_results = pl.read_parquet(pairwise_results_path)
    else:
        pairwise_results = to_pairwise(load_vhelm_results_long())
        pairwise_results.write_parquet(pairwise_results_path)
    return pairwise_results


def load_arena_results():
    """Load the LLM evaluation results."""
    path = ARENA_DIR + "arena.parquet"
    battles = pd.read_parquet(path).sort_values(ascending=True, by=["tstamp"])

    battles = battles[battles["anony"]]
    # battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    return pl.DataFrame(battles)


def get_models(df):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    return models.tolist()


def main():
    results_pairwise_vhelm = load_vhelm_results_pairwise().to_pandas()
    results_pairwise_arena = load_arena_results().to_pandas()

    print(results_pairwise_vhelm.shape)
    print(results_pairwise_arena.shape)

    results_pairwise_vhelm = results_pairwise_vhelm[["model_a", "model_b", "winner"]]
    results_pairwise_arena = results_pairwise_arena[["model_a", "model_b", "winner"]]

    vhelm_models = get_models(results_pairwise_vhelm)
    # print(vhelm_models)

    arena_models = get_models(results_pairwise_arena)
    # print(arena_models)

    # matched_models = {}
    # for model in vhelm_models:
    #     model_name = "_".join(model.split("_")[1:])
    #     match, score, match_index = process.extractOne(
    #         model_name, arena_models, scorer=fuzz.WRatio
    #     )
    #     if score > 74:
    #         matched_models[model] = match
    # print(matched_models)
    matched_models = {
        "google_gemini_pro_vision": "gemini-pro-vision",
        "anthropic_claude_3_opus_20240229": "claude-3-opus",
        "huggingfacem4_idefics2_8b": "idefics2-8b-chatty",
        "anthropic_claude_3_haiku_20240307": "claude-3-haiku",
        "anthropic_claude_3_sonnet_20240229": "claude-3-sonnet",
        "microsoft_llava_1.5_13b_hf": "llava-v1.5-13b",
        "openai_gpt_4_1106_vision_preview": "gpt-4-vision-preview",
        "openai_gpt_4o_2024_05_13": "gpt-4o",
        "openai_gpt_4_vision_preview": "gpt-4-vision-preview",
        "llava_1.6_vicuna_7b": "llava-v1.6-vicuna-7b",
        "llava_1.6_vicuna_13b": "llava-v1.6-vicuna-13b",
        "qwen_qwen_vl_chat": "Qwen-VL-Chat",
    }
    # rename models in the helm results
    results_pairwise_vhelm["model_a"] = results_pairwise_vhelm["model_a"].apply(
        lambda x: matched_models.get(x, x)
    )
    results_pairwise_vhelm["model_b"] = results_pairwise_vhelm["model_b"].apply(
        lambda x: matched_models.get(x, x)
    )

    # compute MLE for Arena
    # drop models not present in the helm results
    results_pairwise_arena = results_pairwise_arena[
        results_pairwise_arena["model_a"].isin(matched_models.values())
        & results_pairwise_arena["model_b"].isin(matched_models.values())
    ]
    ranking_arena = compute_mle(results_pairwise_arena)
    # print(results_pairwise_arena.shape)

    # compute joint MLE for Helm and Arena
    results_pairwise_helm = results_pairwise_vhelm[
        results_pairwise_vhelm["model_a"].isin(matched_models.values())
        & results_pairwise_vhelm["model_b"].isin(matched_models.values())
    ]
    # print(results_pairwise_helm.shape)
    results_pairwise_joint = pd.concat([results_pairwise_helm, results_pairwise_arena])
    ranking_joint = compute_mle(results_pairwise_joint)

    print(ranking_arena)
    print(ranking_joint)


if __name__ == "__main__":
    main()
