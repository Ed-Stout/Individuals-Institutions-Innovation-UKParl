import pandas as pd

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_vocab.csv"
speeches = pd.read_csv(input_csv)

#======split id into parts
id_dates = []
id_letters = []
id_colnums = []
id_seqs = []
bad_ids = []

for speech_id in speeches['id'].astype(str):
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

speeches['id_date'] = id_dates
speeches['id_letter'] = id_letters
speeches['id_colnum'] = id_colnums
speeches['id_seq'] = id_seqs

# ====== the sort itself ======
speeches = speeches.sort_values(['id_date', 'id_colnum', 'id_seq']).reset_index(drop=True)