import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from analysis.elo import (
    pretty_print_model_ratings,
    pretty_print_two_ratings,
    compute_elo_vlm,
)


# Define the Bradley-Terry log-likelihood function
def bradley_terry_log_likelihood(params):
    ll = 0
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                ll += wins[i, j] * (
                    params[i] - np.log(np.exp(params[i]) + np.exp(params[j]))
                )
                ll += losses[i, j] * (
                    params[j] - np.log(np.exp(params[i]) + np.exp(params[j]))
                )
    return -ll


# Load the pairwise comparison data
df = pd.read_parquet("results/pairwise_num/coco2017_cap_val_tie.parquet")
df.set_index(df.columns[0], inplace=True)

model_pairs = df.index.tolist()
all_pairs_list = list(set(model_pairs))

online_elo_ratings = compute_elo_vlm(df)
print(pretty_print_model_ratings(online_elo_ratings))

elo_mle_ratings_reverse = compute_elo_vlm(df.iloc[::-1])
print(
    pretty_print_two_ratings(
        online_elo_ratings,
        elo_mle_ratings_reverse,
        column_names=["Elo rating", "Elo rating with reverse order"],
    )
)

win_count = defaultdict(int)
loss_count = defaultdict(int)

# Count wins and losses
for index, row in df.iterrows():
    model_pair = index.split("_vs_")
    model_a = model_pair[0]
    model_b = model_pair[1]

    for col in df.columns:
        if row[col] == 1:
            win_count[model_a] += 1
            loss_count[model_b] += 1

        elif row[col] == 0:
            win_count[model_b] += 1
            loss_count[model_a] += 1

        elif row[col] == 0.5:
            win_count[model_a] += 0.5
            loss_count[model_a] += 0.5
            win_count[model_b] += 0.5
            loss_count[model_b] += 0.5

# Convert counts to DataFrame
model_names = list(set(win_count.keys()).union(set(loss_count.keys())))
win_data = {model: win_count[model] for model in model_names}
loss_data = {model: loss_count[model] for model in model_names}

win_df = pd.DataFrame.from_dict(win_data, orient="index", columns=["Wins"])
loss_df = pd.DataFrame.from_dict(loss_data, orient="index", columns=["Losses"])

combined_df = win_df.join(loss_df).fillna(0).astype(int)

print("Combined DataFrame with wins and losses:")
print(combined_df)

n_models = len(model_names)
model_index = {model: i for i, model in enumerate(model_names)}

# Prepare win-loss matrices
wins = np.zeros((n_models, n_models))
losses = np.zeros((n_models, n_models))

for index, row in df.iterrows():
    model_pair = index.split("_vs_")
    model_a = model_index[model_pair[0]]
    model_b = model_index[model_pair[1]]

    for col in df.columns:
        if row[col] == 1:
            wins[model_a, model_b] += 1
            losses[model_b, model_a] += 1
        elif row[col] == 0:
            wins[model_b, model_a] += 1
            losses[model_a, model_b] += 1

        elif row[col] == 0.5:
            wins[model_a, model_b] += 0.5
            wins[model_b, model_a] += 0.5
            losses[model_a, model_b] += 0.5
            losses[model_b, model_a] += 0.5

# Initial guess for model strengths
initial_guess = np.zeros(n_models)


# Minimise NLL
result = minimize(bradley_terry_log_likelihood, initial_guess, method="BFGS")
strengths = result.x

strength_df = pd.DataFrame({"Model": model_names, "Strength": strengths}).sort_values(
    by="Strength", ascending=False
)

print("Estimated Model Strengths:")
print(strength_df)
