import numpy as np
import pandas as pd
import os

scales = [60,250,1000,7500] #non-chair speeches - x8.5 more than Barron

window_excluded_tiers = ["chair"]
all_tiers = ["chair", "government", "opposition", "backbencher"]

source = r"G:\My Drive\Birkbeck\Project\Hansard"
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
order_base = ordered_corpus[0]
expected_order = np.arange(order_base, order_base + len(corpus))

row_sums = mixtures.sum(axis=1) #sums to one across topics
print("row sums max:", row_sums.max())
print("row sums min:", row_sums.min())

print(corpus["role_tier"].value_counts())

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

if not corpus["date"].is_monotonic_increasing:
    raise SystemExit("dates not in order!")

if mixtures.min() <= 0:
    print("smallest probability:", mixtures.min())
    raise SystemExit("Zero probability") #KLD doesn't work on zeroes

for tier in all_tiers: #added after issue
    if tier not in corpus["role_tier"].unique():
        print(corpus["role_tier"].value_counts())
        raise SystemExit("missing role tier, check spelling")

#=========remove chairs ======================================
role_tiers = corpus["role_tier"].tolist()

keep = []

for tier in role_tiers:
    if tier in window_excluded_tiers:
        keep.append(False) #chair gets dropped
    else:
        keep.append(True)

keep = np.array(keep) #needs to be array to filter

print("dropping", len(corpus) - keep.sum(), "chair speeches") #cnt

corpus = corpus[keep].reset_index(drop=True)
mixtures = mixtures[keep]
if len(corpus) != mixtures.shape[0]:
    raise SystemExit("corpus and mixtures out of step after filtering!")

for scale in scales:
    if 2 * scale >= len(corpus):
        raise SystemExit("scale " + str(scale) + " too big for corpus")

corpus["original_analysis_order"] = corpus["analysis_order"] #keep original order
corpus["analysis_order"] = np.arange(len(corpus))

#checks
print("corpus now:", corpus.shape)
print(corpus["role_tier"].value_counts())

#=========scoreable range=================
#first and last w speeches have nothing on one side
def scoreable_range(scale):
    speech_start = scale
    speech_end = len(corpus) - scale
    return speech_start, speech_end

for scale in scales:
    start, end = scoreable_range(scale)
    print("scale", scale, "| scoreable", end - start, "of", len(corpus))

