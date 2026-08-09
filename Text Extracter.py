import pandas as pd

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-v310.csv"

all_speeches = pd.read_csv(input_csv)
all_speeches_2016 = all_speeches[all_speeches["year"] == 2016] #use only 2016 data to begin with
all_speeches_2016.to_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016.csv", index=False)

#print(f"Number of speeches in 2016: {len(all_speeches_2016)}")
print(all_speeches_2016["date"].min(), all_speeches_2016["date"].max())