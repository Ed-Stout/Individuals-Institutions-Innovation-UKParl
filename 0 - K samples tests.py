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
#sample_size = 50000
n_iter = 1500
alpha = 0.1
eta = 0.01
random_state = 42
topicnum = 100

#=======evenly spaced sample, not the first n=========
all_texts = []
with open(output_path / 'corpus.txt', encoding='utf-8') as f:
    for line in f:
        all_texts.append(line)

step = 7 #Proportional to sample size

texts = []
for position in range(0, len(all_texts), step):
    texts.append(all_texts[position])
#texts = texts[:sample_size]

print("documents in sample:", len(texts)) #sample size

CVzer = CountVectorizer(token_pattern=r"(?u)\S+", max_features=None) #whitespaces, vocab already capped and lowercased
doc_vcnts = CVzer.fit_transform(texts)
vocabulary = CVzer.get_feature_names_out()

zero_rows = (doc_vcnts.sum(axis=1) == 0).sum() #check no zeroes

print("vocabulary:", len(vocabulary)) #check numbers of words
print("all-zero rows:", zero_rows) 

#==========fit============
start = time.time()

lda_model = lda.LDA(n_topics=topicnum, n_iter=n_iter, refresh=100, alpha=alpha, eta=eta, random_state=random_state) #parameters
lda_model.fit(doc_vcnts)

elapsed = time.time() - start
print("elapsed mins:", round(elapsed / 60, 1))

topic_word = lda_model.topic_word_