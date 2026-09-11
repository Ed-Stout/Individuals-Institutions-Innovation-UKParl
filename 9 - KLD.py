#9 - KLD. Novelty, transience, and resonance calculated across the whole corpus
import numpy as np
import pandas as pd
import os

scales = [1,60,250,1000,7500] 
chunk_size = 25000 #centres per block - whole-array version runs out of memory
window_excluded_tiers = ["chair"]
all_tiers = ["chair", "government", "opposition", "backbencher"]

source = r"G:\My Drive\Birkbeck\Project\Hansard"
lda_output = r"C:\Dissertation Project\LDA_output" #script 8 saves here, not to Drive
topic_mixtures_npy = os.path.join(lda_output, "topic_mixtures_k100.npy")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
output_csv = os.path.join(source, "hansard_ntr_by_scale.csv")

#=============load topics=================
mixtures = np.load(topic_mixtures_npy)

print("topic mixtures:", mixtures.shape)
print("size:", round(mixtures.nbytes / 1e6, 1), "MB") #size

use_columns = ["id", "analysis_order", "speech_order", "date", "role_tier"] #only what this script reads
corpus = pd.read_csv(corpus_csv, usecols=use_columns, parse_dates=["date"]) #
print("corpus loaded:", corpus.shape)

corpus = corpus.sort_values("analysis_order").reset_index(drop=True)#align 

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

for scale in scales: #scale 0 would divide by nothing
    if scale < 1:
        raise SystemExit("scale below 1: " + str(scale))

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

corpus["original_analysis_order"] = corpus["analysis_order"] #joins back to the corpus
corpus["kld_row"] = np.arange(len(corpus))  
corpus = corpus.drop(columns=["analysis_order"]) 

print("corpus now:", corpus.shape)
print(corpus["role_tier"].value_counts())

#=========scoreable range=================
def scoreable_range(scale):
    speech_start = scale
    speech_end = len(corpus) - scale
    return speech_start, speech_end

for scale in scales:
    start, end = scoreable_range(scale)
    print("scale: ", scale, "scoreable: ", end - start, "of: ", len(corpus))

#=========running totals=======================
log_mixtures = np.log2(mixtures) #how spread out each speech is across the 100 topics
weighted = mixtures * log_mixtures #each topic's probability times its own log
entropy = -weighted.sum(axis=1) #add up each row and make it positive, one per speech

del weighted

n_speeches = len(corpus)
n_topics = mixtures.shape[1]  #(rows, columns)

cumulative = np.zeros((n_speeches + 1, n_topics)) #goes down the rows
np.cumsum(log_mixtures, axis=0, out=cumulative[1:]) #axis=0 accumulates down the speeches

print("running totals built:", cumulative.shape)
print("memory:", round(cumulative.nbytes / 1e6, 1), "MB") #added because of many errors

del log_mixtures

def novelty_transience_resonance(scale):
    speech_start, speech_end = scoreable_range(scale)

    centres = np.arange(speech_start, speech_end)
    novelty = np.zeros(len(centres))
    transience = np.zeros(len(centres))

    for chunk_start in range(0, len(centres), chunk_size): #chunked to keep temporaries 
        chunk_stop = min(chunk_start + chunk_size, len(centres))
        chunk_centres = centres[chunk_start:chunk_stop]

        past_start = chunk_centres - scale #window edges as row numbers
        past_stop = chunk_centres
        future_start = chunk_centres + 1
        future_stop = chunk_centres + scale + 1

        past_mean = (cumulative[past_stop] - cumulative[past_start]) / scale #mean of log2(mixture) across each window, read off the running totals
        future_mean = (cumulative[future_stop] - cumulative[future_start]) / scale

        centre_mixtures = mixtures[chunk_centres]
        centre_entropy = entropy[chunk_centres]

        novelty[chunk_start:chunk_stop] = -centre_entropy - (centre_mixtures * past_mean).sum(axis=1) #KLD averaged over a window = -entropy(centre) - dot(centre, window mean)
        transience[chunk_start:chunk_stop] = -centre_entropy - (centre_mixtures * future_mean).sum(axis=1)

    resonance = novelty - transience
    return centres, novelty, transience, resonance

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

    print("Scale: ", scale, "Scored: ", len(centres),"Novelty: ", round(novelty.mean(), 3),"Transience: ", round(transience.mean(), 3),"Resonance: ", round(resonance.mean(), 4))

results.to_csv(output_csv, index=False)
print("saved:", output_csv)
print(results.shape)