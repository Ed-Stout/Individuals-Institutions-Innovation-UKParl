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

#=========running totals=======================
log_mixtures = np.log2(mixtures)
#how spread out each speech is across the 100 topics
weighted = mixtures * log_mixtures #each topic's probability times its own log

entropy = -weighted.sum(axis=1) # one per speech

del weighted

n_speeches = len(corpus)
n_topics = mixtures.shape[1]

cumulative = np.zeros((n_speeches + 1, n_topics)) #goes down the rows

cumulative[1:] = np.cumsum(log_mixtures, axis=0) #axis=0 accumulates down the speeches

print("running totals built:", cumulative.shape)
print("memory:", round(cumulative.nbytes / 1e6, 1), "MB")

"""#=========Barron's original, for checking

def barron_kld(pdists0, pdists1):
    return (pdists1 * np.log2(pdists1 / pdists0)).sum(axis=1)

def barron_ntr(thetas_arr, scale):
    speechstart = scale
    speechend = thetas_arr.shape[0] - scale

    novelties = []
    transiences = []
    resonances = []

    for j in range(speechstart, speechend, 1):
        center_theta = thetas_arr[j]

        after_boxend = j + scale + 1
        before_boxstart = j - scale

        before_theta_arr = thetas_arr[before_boxstart:j]
        beforenum = before_theta_arr.shape[0]
        before_centertheta_arr = np.tile(center_theta, reps=(beforenum, 1))

        after_theta_arr = thetas_arr[j + 1:after_boxend]
        afternum = after_theta_arr.shape[0]
        after_centertheta_arr = np.tile(center_theta, reps=(afternum, 1))

        before_KLDs = barron_kld(before_theta_arr, before_centertheta_arr)
        after_KLDs = barron_kld(after_theta_arr, after_centertheta_arr)

        novelty = np.mean(before_KLDs)
        transience = np.mean(after_KLDs)

        novelties.append(novelty)
        transiences.append(transience)
        resonances.append(novelty - transience)

    return np.array(novelties), np.array(transiences), np.array(resonances)"""

def novelty_transience_resonance(scale):
    speech_start, speech_end = scoreable_range(scale)

    centres = np.arange(speech_start, speech_end)

        #window edges as row numbers
    past_start = centres - scale
    past_stop = centres
    future_start = centres + 1
    future_stop = centres + scale + 1