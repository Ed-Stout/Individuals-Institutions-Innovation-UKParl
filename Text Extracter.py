import pandas as pd

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-v310.csv"

df_chunk = []
chunk_cnt = 0

for chunk in pd.read_csv(input_csv, chunksize=100000, dtype=str):
    df_chunk.append(chunk[chunk["year"] == "2016"])  # Filter for year 2016
    chunk_cnt += 1
    print("chunk", chunk_cnt, "done")

all_speeches_2016 = pd.concat(df_chunk)#use only 2016 data to begin with
all_speeches_2016.to_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016.csv", index=False)

#print(f"Number of speeches in 2016: {len(all_speeches_2016)}")
print(all_speeches_2016["date"].min(), all_speeches_2016["date"].max())
print(f"kept {len(all_speeches_2016)} speeches from 2016")