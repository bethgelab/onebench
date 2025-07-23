import polars as pl
import pandas as pd
import choix
from elo import compute_mle, to_pairwise, compute_elo, compute_tau, compute_plackett_luce_pairwise
from lsr import *
from utils import *
import random
from robustness import load_long, rank_models, compute_pairwise

root_dir = ''
def load_vlm_results(dataset):
    if dataset == 'vhelm':
        results_path = f'{root_dir}data/vlm/vhelm/results.parquet'
    else:
        results_path = f'{root_dir}data/vlm/lmms-eval/results_full_data.parquet'
    return pl.read_parquet(results_path)

print("Loading parquet")
dataset='lmms-eval'
results = load_vlm_results(dataset)

ranking = rank_models(results)
print(results)

columns = results.columns
sampled_columns = random.sample(columns, int( len(columns)))
sampled = results.select(sampled_columns)
if "model" not in sampled.columns:
    sampled = sampled.with_columns(results["model"].alias("model"))
long_sampled = load_long(sampled)
pairwise_sampled = to_pairwise(long_sampled, long=True)

ranking_mle = compute_mle(pairwise_sampled.to_pandas())
ranking_elo = compute_elo(pairwise_sampled.to_pandas())
print("MLE, ELO done")



tau_mle = compute_tau(ranking["model"], ranking_mle.index)
tau_elo = compute_tau(ranking["model"], ranking_elo.index)

pair_list, models = compute_pairwise(pairwise_sampled)
bt_params = choix.lsr_pairwise(len(models), pair_list, alpha=0.05)
bt_rank = pd.Series(bt_params, index=models).sort_values(ascending=False)

sampled = sampled.to_pandas()
sampled.set_index("model", inplace=True)
top1_ranks = gen_top1_list(sampled)
lsr_params = choix.lsr_top1(len(sampled.index), top1_ranks, alpha=0.05)
lsr_rank = pd.Series(lsr_params, index=sampled.index).sort_values(ascending=False)

tau_bt = compute_tau(ranking["model"], bt_rank.index)
tau_lsr = compute_tau(ranking["model"], lsr_rank.index)

print("BT, PL done")

print("LSR:", tau_lsr)
print("Ours: ", tau_bt)
print("LMArena:", tau_mle)
print("ELO:", tau_elo)
print()