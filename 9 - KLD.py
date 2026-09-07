import numpy as np
import pandas as pd
import os

scales = [1,60,250,1000,7500] #non-chair speeches - x8.5 more than Barron

chair_in_windows = False  #False follows Barron SI 2.4 - chairs contribute nothing to a window
chair_as_centre = True    #True scores chairs as centres, so the chair tier reaches script 12

chair_tier = "chair"
all_tiers = ["chair", "government", "opposition", "backbencher"]

source = r"G:\My Drive\Birkbeck\Project\Hansard"
topic_mixtures_npy = os.path.join(source, "topic_mixtures_k100.npy")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
output_csv = os.path.join(source, "hansard_ntr_by_scale.csv")

for scale in scales: #scale 0 would divide by nothing
    if scale < 1:
        raise SystemExit("scale below 1: " + str(scale))

#=============load topics=================
mixtures = np.load(topic_mixtures_npy)

print("topic mixtures:", mixtures.shape)
print("size:", round(mixtures.nbytes / 1e6, 1), "MB") #size

use_columns = ["id", "analysis_order","speech_order", "date",  "display_as","person_id", "party",
               "role", "role_tier", "major_heading", "minor_heading"]

corpus = pd.read_csv(corpus_csv, usecols=use_columns, parse_dates=["date"]) #
print("corpus loaded:", corpus.shape)

corpus = corpus.sort_values("analysis_order").reset_index(drop=True)#align 

ordered_corpus = corpus["analysis_order"].to_numpy() #get rid
order_base = ordered_corpus[0]
expected_order = np.arange(order_base, order_base + len(corpus)) #to check against

row_sums = mixtures.sum(axis=1) #sums to one across topics
print("row sums max:", row_sums.max())
print("row sums min:", row_sums.min())

if not np.allclose(row_sums, 1.0):
    raise SystemExit("topic mixtures don't sum to one")

print(corpus["role_tier"].value_counts())

#=================checks======================

if len(corpus) != mixtures.shape[0]:
    print("corpus rows:", len(corpus))
    print("mixtures rows:", mixtures.shape[0])
    raise SystemExit("mixtures and topic mixture row counts differ")

if not np.array_equal(ordered_corpus, expected_order): #get rid
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
role_tiers = corpus["role_tier"].to_numpy()

in_window = []
is_centre = []

for tier in role_tiers:
    if tier == chair_tier and not chair_in_windows:
        in_window.append(0.0)
    else:
        in_window.append(1.0)

    if tier == chair_tier and not chair_as_centre:
        is_centre.append(False)
    else:
        is_centre.append(True)

in_window = np.array(in_window)
is_centre = np.array(is_centre)

n_speeches = len(corpus)

print("contributing to windows:", int(in_window.sum()), "of", n_speeches)
print("eligible as centres:   ", int(is_centre.sum()), "of", n_speeches)

if in_window.sum() == 0:
    raise SystemExit("every speech masked out of windows, check chair_tier spelling")

block_id = np.zeros(n_speeches, dtype=int)

for break_date in parliament_breaks:
    after_break = corpus["date"] > = pd.Timestamp(break_date)
    block_id = block_id + after_break.to_numpy().astype(int)

    first_row = int(np.argmax(after_break.to_numpy()))
    print("break", break_date, "first row: ", first_row, "last sitting: ", str(corpus["date"].iloc[first_row - 1])[:10])
keep = np.array(keep) #needs to be array to filter

print("dropping", len(corpus) - keep.sum(), "chair speeches") #cnt

corpus = corpus[keep].reset_index(drop=True)
mixtures = mixtures[keep]
if len(corpus) != mixtures.shape[0]:
    raise SystemExit("corpus and mixtures out of step after filtering!")

for scale in scales: #get rid
    if 2 * scale >= len(corpus):
        raise SystemExit("scale " + str(scale) + " too big for corpus")

corpus["original_analysis_order"] = corpus["analysis_order"]   #the real key - joins back to the corpus
corpus["kld_row"] = np.arange(len(corpus))                     #position in the chair-free sequence, for reading only
corpus = corpus.drop(columns=["analysis_order"])               #don't ship two different meanings under one name

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

#=========running totals=======================
log_mixtures = np.log2(mixtures) #how spread out each speech is across the 100 topics
weighted = mixtures * log_mixtures #each topic's probability times its own log
entropy = -weighted.sum(axis=1) #add up each row and make it positive, one per speech

del weighted

n_speeches = len(corpus)
n_topics = mixtures.shape[1]  #(rows, columns)

cumulative = np.zeros((n_speeches + 1, n_topics)) #goes down the rows
cumulative[1:] = np.cumsum(log_mixtures, axis=0) #axis=0 accumulates down the speeches

print("running totals built:", cumulative.shape)
print("memory:", round(cumulative.nbytes / 1e6, 1), "MB")

del log_mixtures

def novelty_transience_resonance(scale):
    speech_start, speech_end = scoreable_range(scale)

    centres = np.arange(speech_start, speech_end)

    #window edges as row numbers
    past_start = centres - scale
    past_stop = centres
    future_start = centres + 1
    future_stop = centres + scale + 1

#mean of log2(mixture) across each window, read off the running totals
    past_mean = (cumulative[past_stop] - cumulative[past_start]) / scale
    future_mean = (cumulative[future_stop] - cumulative[future_start]) / scale

    centre_mixtures = mixtures[centres]
    centre_entropy = entropy[centres]

    #KLD averaged over a window = -entropy(centre) - dot(centre, window mean)
    novelty = -centre_entropy - (centre_mixtures * past_mean).sum(axis=1)
    transience = -centre_entropy - (centre_mixtures * future_mean).sum(axis=1)
    resonance = novelty - transience

    return centres, novelty, transience, resonance

#=========quick check on one scale
centres, novelty, transience, resonance = novelty_transience_resonance(scales[0])

print("scale", scales[0])
print("scored:", len(centres))
print("novelty     ", round(novelty.mean(), 3), "| range", round(novelty.min(), 3), "to", round(novelty.max(), 3))
print("transience  ", round(transience.mean(), 3))
print("resonance   ", round(resonance.mean(), 4))

#=========run every scale=============
results = corpus[["kld_row", "original_analysis_order", "id", "date"]].copy()

for scale in scales:
    centres, novelty, transience, resonance = novelty_transience_resonance(scale)

    results["novelty_" + str(scale)] = np.nan #speeches without a full window stay as blanks
    results["transience_" + str(scale)] = np.nan
    results["resonance_" + str(scale)] = np.nan

    results.loc[centres, "novelty_" + str(scale)] = novelty
    results.loc[centres, "transience_" + str(scale)] = transience
    results.loc[centres, "resonance_" + str(scale)] = resonance

    print("scale ", scale, "scored ", len(centres),
          "novelty ", round(novelty.mean(), 3),
          "transience ", round(transience.mean(), 3),
          "resonance ", round(resonance.mean(), 4))

results.to_csv(output_csv, index=False)
print("saved:", output_csv)
print(results.shape)