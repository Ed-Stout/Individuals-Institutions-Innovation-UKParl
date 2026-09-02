import numpy as np
import pandas as pd
import os

scales = [60,250,1000,7500] #non-chair speeches - x8.5 more than Barron

window_excluded_tiers = ["chair"]
all_tiers = ["chair", "government", "opposition", "backbencher"]

min_probability = 
sum_to_one = 

source = r"G:\My Drive\Birkbeck\Project\Hansard\LDA_output"
topic_mixtures_npy = os.path.join(source, "topic_mixtures_k100.npy")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
output_csv = os.path.join(source, "hansard_ntr_by_scale.csv")

#=========load topics
mixtures = np.load(topic_mixtures_npy)

print("topic mixtures:", mixtures.shape)
print("size:", round(mixtures.nbytes / 1e6, 1), "MB") #size

use_columns = ["id", "analysis_order","speech_order", "date",  "display_as","person_id", "party",
               "role", "role_tier", "major_heading", "minor_heading"]

corpus = pd.read_csv(corpus_csv, usecols=use_columns, parse_dates=["date"]) #
print("corpus loaded:", corpus.shape)

corpus = corpus.sort_values("analysis_order").reset_index(drop=True)

ordered_corpus = corpus["analysis_order"].to_numpy()
order_base = order_values[0]
expected_order = np.arange(order_base, order_base + len(corpus))

row_sums = mixtures.sum(axis=1) #sums to one across topics
print("row sums max:", row_sums.max())
print("row sums min:", row_sums.min())

print(corpus["role_tier"].value_counts())

#is_measurable = ~corpus

#=================checks======================

if len(corpus) != mixtures.shape[0]:
    print("corpus rows:", len(corpus))
    print("mixtures rows:", mixtures.shape[0])
    raise SystemExit("mixtures and topic mixture row counts differ")

if not np.array_equal(ordered_corpus, expected_order):
    print("first five values:", ordered_corpus[:5])
    print("duplicates:", len(ordered_corpus) - len(set(ordered_corpus))) 
    raise SystemExit("analysis is not in order!")

if not corpus["speech_order"].is_monotonic_increasing:
    raise SystemExit("speeches not in order!")

if not corpus["date"].is_monotic_increasing:
    raise SystemExit("dates not in order!")

role_tiers = corpus["role_tier"].tolist()
is_measurable = []
for tier in role_tiers:
    if tier in window_excluded_tiers:
        is_measurable.append(False) 
    else:
        is_measurable.append(True) #can appear in anpother window
is_measurable = np.array(is_measurable)
print("measurable:", is_measurable.sum(), "of", len(corpus))

for scale in scales:
    if 2 * scale >= is_measurable.sum():
        raise SystemExit("Scale too big for corpus")

measurable_rows = np.flatnonzero(is_measurable) #ordered row num of every speech in a window
n_measurable = len(measurable_rows) #number of rows measurable

measurable_cnt = np.cumsum(is_measurable) #running count of speeches

measurable_before = measurable_cnt - is_measurable #before, incl minus its own count