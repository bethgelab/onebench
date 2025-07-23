import pandas as pd
import plotly.express as px


def visualize_pairwise_comparisons(battles, title, show_num_models=30):
    """Visualize the number of battles between models."""
    ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )
    battle_counts = ptbl + ptbl.T
    ordering = battle_counts.sum().sort_values(ascending=False).index
    ordering = ordering[:show_num_models]
    fig = px.imshow(battle_counts.loc[ordering, ordering], title=title, text_auto=True)
    fig.update_layout(
        xaxis_title="Model B",
        yaxis_title="Model A",
        xaxis_side="top",
        height=800,
        width=800,
        title_y=0.07,
        title_x=0.5,
        font=dict(size=10),
    )
    fig.update_traces(
        hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Count: %{z}<extra></extra>"
    )
    return fig
