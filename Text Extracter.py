import pandas as pd

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-v310.csv"

kept_chunks = []
chunk_cnt = 0
years = ["2014", "2016", "2018", "2020"]

for chunk in pd.read_csv(input_csv, chunksize=100000, dtype=str):
    matching_rows = chunk[chunk["year"].isin(years)]
    if len(matching_rows) > 0:              # skip empties, keeps concat clean
        kept_chunks.append(matching_rows)
    chunk_cnt += 1
    print("chunk", chunk_cnt, "done")


all_speeches = pd.concat(kept_chunks)
all_speeches.to_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-two-year-intervals.csv", index=False)

#print(f"Number of speeches in 2016: {len(all_speeches_2016)}")
print(all_speeches["date"].min(), all_speeches["date"].max())
print(f"kept {len(all_speeches)} speeches")