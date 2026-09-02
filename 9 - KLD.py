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

row_sums = mixtures.sum(axis=1)
print("row sums max:", row_sums.max())
print("row sums min:", row_sums.min())

print(corpus["role_tier"].value_counts())

#is_measurable = ~corpus



