import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from analysis import compute_plackett_luce_pairwise_dense
from analysis import load_pairwise_results

# Set style
plt.style.use(["science", "vibrant", "grid"])
plt.rcParams["font.family"] = "Times"

true_parameters = pd.read_csv("data/llm/synthetic/model_parameters.csv", index_col=0)
pairwise_results = load_pairwise_results(benchmark="synthetic")
pl_ranking = compute_plackett_luce_pairwise_dense(pairwise_results)


# Sort by model number
def extract_model_number(model_name):
    return int("".join(filter(str.isdigit, model_name)))


pl_ranking = pl_ranking.sort_index(key=lambda x: x.map(extract_model_number))

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()

# Plot data
x = ["True Parameters", "PL Scores"]

# Plot points
ax1.plot([0] * len(true_parameters["mean"]), true_parameters["mean"], "o", alpha=0.5)
ax2.plot([1] * len(pl_ranking), pl_ranking, "o", alpha=0.5)

print(len(true_parameters["mean"]), len(pl_ranking))
print(true_parameters["mean"])

ax1.set_ylim(min(true_parameters["mean"]), max(true_parameters["mean"]))
ax2.set_ylim(min(pl_ranking), max(pl_ranking))

# Labels
ax1.set_ylabel("True Parameters")
ax2.set_ylabel("PL Scores")

# Save plot
plt.tight_layout()
plt.savefig("figures/gaps.pdf", bbox_inches="tight")
