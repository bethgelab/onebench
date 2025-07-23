import pandas as pd
import os
import glob
import random

# Path where parquet files are stored
type = 'binary'
source = 'lmms-eval'
parquet_files_path = f'/Users/heikekoenig/irp/lifelong_analysis/data/vlm/{source}/{type}'
output_path = f'/Users/heikekoenig/irp/lifelong_analysis/data/vlm/{source}/{type}.parquet'

parquet_files = glob.glob(os.path.join(parquet_files_path, '*.parquet'))
print(len(parquet_files))

count = 0
dataframes = []
final_num = []

for file in parquet_files:
    print(file)
    df = pd.read_parquet(file)
    print(df)
    print(df.shape[1])
    if type == 'binary':
        count += df.shape[1]
        dataframes.append(df)
    else:
        columns = list(df.columns)
        # columns.remove('model')
        sample_size = len(columns) // 2
        selected_items = random.sample(columns, sample_size)

        final_num.extend(selected_items)
        count += len(selected_items)
        # selected_items.insert(0, 'model')
        dataframes.append(df[selected_items])



# dataframes = [pd.read_parquet(file) for file in parquet_files]

# Combine all the DataFrames along the columns
combined_df = pd.concat(dataframes, axis=1)
print(len(combined_df.columns))
# Replace NaN with None
combined_df = combined_df.where(pd.notnull(combined_df), None)

combined_df = combined_df.reset_index()
combined_df = combined_df.rename(columns={'index': 'model'})
combined_df.to_parquet(output_path, compression='snappy')
print(combined_df)

print("count is ", count)
print(f"Combined DataFrame saved to {output_path}")

if type == 'numeric':
    # Save the selected numeric features to a txt file
    with open(f'/Users/heikekoenig/irp/lifelong_analysis/data/vlm/{source}/selected_numeric_features.txt', 'w') as f:
        for item in final_num:
            f.write("%s\n" % item)
