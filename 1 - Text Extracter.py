import pandas as pd

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\Source data\hansard-speeches-v310.csv"
output_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016_20.csv"
order_csv = r"G:\My Drive\Birkbeck\Project\Hansard\speech_order_2016_20.csv"

kept_chunks = []
chunk_cnt = 0
years = ["2015", "2016", "2017", "2018", "2019", "2020"]
#years = ["2016"]

for chunk in pd.read_csv(input_csv, chunksize=100000, dtype=str):
    matching_rows = chunk[chunk["year"].isin(years)]
    if len(matching_rows) > 0:              # skip empties, keeps concat clean
        kept_chunks.append(matching_rows)
    chunk_cnt += 1
    print("chunk", chunk_cnt, "done")

all_speeches = pd.concat(kept_chunks)

#======split id into parts. From id splitter.py
id_dates = []
id_letters = []
id_colnums = []
id_seqs = []
bad_ids = []

for speech_id in all_speeches['id'].astype(str):
    tail = speech_id.split('/')[-1]
    parts = tail.split('.') # date, colnum, minicolnum

    if len(parts) != 3: #should be three sections
        bad_ids.append(speech_id)
        id_dates.append(None)
        id_letters.append(None)
        id_colnums.append(-1)
        id_seqs.append(-1)
        continue

    date_letter = parts[0]
    id_dates.append(date_letter[:10])   # "2016-01-05"
    id_letters.append(date_letter[10:]) # "d"
    id_colnums.append(int(parts[1]))        # 1
    id_seqs.append(int(parts[2]))           # 2

all_speeches['id_date'] = id_dates
all_speeches['id_letter'] = id_letters
all_speeches['id_colnum'] = id_colnums
all_speeches['id_seq'] = id_seqs

# ====== sort into the chronological order, then number ======
all_speeches = all_speeches.sort_values(["id_date", "id_colnum", "id_seq"])
all_speeches = all_speeches.reset_index(drop=True) #allows for order later

speech_order = []
for position in range(len(all_speeches)): #give position
    speech_order.append(position)

all_speeches["speech_order"] = speech_order

print(all_speeches["date"].min(), all_speeches["date"].max())
print(f"kept {len(all_speeches)} speeches")

all_speeches.to_csv(output_csv, index=False)

order_columns = ["id", "id_date", "id_colnum", "id_seq", "speech_order"]
all_speeches[order_columns].to_csv(order_csv, index=False)