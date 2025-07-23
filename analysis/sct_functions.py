import pandas as pd
import numpy as np

# def borda_count(pairwise_df):
#     models = pd.unique(pairwise_df[['model_a', 'model_b']].values.ravel('K'))
#     scores = pd.Series(0, index=models, name='borda_score')
#     for _, row in pairwise_df.iterrows():
#         winner = row['winner']
#         scores[winner] += 1
#     return scores.sort_values(ascending=False)


def borda_count(pairwise_df):
    print("creating borda count rank")
    pairwise_df.columns = pairwise_df.columns.str.strip()

    # Filter out ties early
    df = pairwise_df[pairwise_df['winner'] != 'tie'].copy()

    # Map 'winner' column from 'model_a'/'model_b' to actual model names
    df['winner_model'] = np.where(df['winner'] == 'model_a', df['model_a'], df['model_b'])
    df['loser_model'] = np.where(df['winner'] == 'model_a', df['model_b'], df['model_a'])

    # Get all unique models
    models = pd.unique(df[['model_a', 'model_b']].values.ravel('K'))

    # Create comparison matrix as a flat dataframe
    comparison_df = df.groupby(['winner_model', 'loser_model']).size().unstack(fill_value=0)

    # Reindex to ensure all models are present
    comparison_df = comparison_df.reindex(index=models, columns=models, fill_value=0)

    # Borda score is total number of victories
    borda_scores = comparison_df.sum(axis=1)

    return borda_scores.sort_values(ascending=False)

def dowdall_score2(pairwise_df):
    print("creating dowdall rank")
    # Get all unique models
    models = pd.unique(pairwise_df[['model_a', 'model_b']].values.ravel('K'))

    # Initialize scores
    dowdall_scores = pd.Series(0.0, index=models, name='dowdall_score')

    # Filter out ties early
    df = pairwise_df[pairwise_df['winner'] != 'tie'].copy()

    # Map 'winner' column from 'model_a'/'model_b' to actual model names
    df['winner_model'] = np.where(df['winner'] == 'model_a', df['model_a'], df['model_b'])
    df['loser_model'] = np.where(df['winner'] == 'model_a', df['model_b'], df['model_a'])

    # Handle ties (split scores evenly for both models in case of a tie)
    tie_df = pairwise_df[pairwise_df['winner'] == 'tie']
    for model_a, model_b in zip(tie_df['model_a'], tie_df['model_b']):
        # dowdall_scores[model_a] += 0.75
        # dowdall_scores[model_b] += 0.75

        continue
    # Process non-tie rows for Dowdall scores
    df['winner_score'] = 1.0
    df['loser_score'] = 0.5

    # Group by winner/loser and add scores
    score_df = df.groupby(['winner_model', 'loser_model']).agg({'winner_score': 'sum', 'loser_score': 'sum'}).reset_index()

    # Update Dowdall scores using aggregated values
    for _, row in score_df.iterrows():
        dowdall_scores[row['winner_model']] += row['winner_score']
        dowdall_scores[row['loser_model']] += row['loser_score']

    return dowdall_scores.sort_values(ascending=False)


def dowdall_score(pairwise_df):
    print("creating dowdall rank")
    # Get all unique models
    models = pd.unique(pairwise_df[['model_a', 'model_b']].values.ravel('K'))

    # Initialize scores
    dowdall_scores = pd.Series(0.0, index=models, name='dowdall_score')

    # Handle ties - vectorized operation
    tie_df = pairwise_df[pairwise_df['winner'] == 'tie']
    if not tie_df.empty:
        # Add 0.75 to both models in tie cases
        tie_contributions = pd.concat([
            pd.Series(0.75, index=tie_df['model_a']),
            pd.Series(0.75, index=tie_df['model_b'])
        ])
        # Group by index to sum up contributions for each model
        tie_contributions = tie_contributions.groupby(level=0).sum()
        dowdall_scores = dowdall_scores.add(tie_contributions, fill_value=0)

    # Process non-tie rows - vectorized approach
    non_tie_df = pairwise_df[pairwise_df['winner'] != 'tie']

    if not non_tie_df.empty:
        # Create winner and loser series directly
        winners = non_tie_df.apply(
            lambda row: row['model_a'] if row['winner'] == 'model_a' else row['model_b'],
            axis=1
        )
        losers = non_tie_df.apply(
            lambda row: row['model_b'] if row['winner'] == 'model_a' else row['model_a'],
            axis=1
        )

        # Add 1.0 to winners
        winner_contributions = winners.value_counts().reindex(models, fill_value=0)
        dowdall_scores = dowdall_scores.add(winner_contributions, fill_value=0)

        # Add 0.5 to losers
        loser_contributions = losers.value_counts().reindex(models, fill_value=0) * 0.5
        dowdall_scores = dowdall_scores.add(loser_contributions, fill_value=0)

    return dowdall_scores.sort_values(ascending=False)


def dowdall_score3(pairwise_df):
    print("creating dowdall rank")
    # Get all unique models
    models = pd.unique(pairwise_df[['model_a', 'model_b']].values.ravel('K'))

    # Create a ballot matrix representation based on pairwise comparisons
    # First we need to convert the pairwise data to a rank-based format

    # Initialize an empty DataFrame to store the ballot (models as columns, comparisons as rows)
    ballot_matrix = pd.DataFrame(index=range(len(pairwise_df)), columns=models)

    # Fill the ballot matrix based on pairwise comparisons
    for idx, row in pairwise_df.iterrows():
        model_a = row['model_a']
        model_b = row['model_b']
        winner = row['winner']

        if winner == 'model_a':
            # model_a ranked higher than model_b
            ballot_matrix.loc[idx, model_a] = 1
            ballot_matrix.loc[idx, model_b] = 2
        elif winner == 'model_b':
            # model_b ranked higher than model_a
            ballot_matrix.loc[idx, model_a] = 2
            ballot_matrix.loc[idx, model_b] = 1
        else:  # tie case
            # Both get the same rank
            ballot_matrix.loc[idx, model_a] = 1.5
            ballot_matrix.loc[idx, model_b] = 1.5

    # Fill NaN values with a lower rank (models not compared in a given row)
    max_rank = 2  # Since we're only comparing pairs
    for i in range(len(ballot_matrix)):
        missing_models = ballot_matrix.iloc[i].isna()
        if missing_models.any():
            ballot_matrix.iloc[i, missing_models] = max_rank + 1

    # Apply the Dowdall method: 1/rank for each position
    # Convert ranks to Dowdall scores (1/rank)
    dowdall_matrix = 1 / ballot_matrix

    # Sum the scores for each model
    dowdall_scores = dowdall_matrix.sum()
    dowdall_scores.name = 'dowdall_score'

    # Return sorted scores (descending)
    return dowdall_scores.sort_values(ascending=False)