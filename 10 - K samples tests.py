#K samples tests
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import lda

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
save_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
save_path.mkdir(parents=True, exist_ok=True)

#=======parameters =========
sample_size = 50000
n_iter = 1500
alpha = 0.1
eta = 0.01
random_state = 42

#=======evenly spaced sample, not the first n=========
all_texts = []
with open(output_path / 'corpus.txt', encoding='utf-8') as f:
    for line in f:
        all_texts.append(line)

step = 7 #Proportional to sample size