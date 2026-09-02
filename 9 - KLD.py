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
thetas = np.load(topic_mixtures_npy)

print("topic mixtures:", thetas)
print("size:", round(thetas.nbytes / 1e6, 1), "MB")
