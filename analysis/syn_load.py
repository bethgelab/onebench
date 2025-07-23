import os
import pandas as pd

#load parquet file
df = pd.read_parquet("data/llm/synthetic/pairwise.parquet")
print(df)