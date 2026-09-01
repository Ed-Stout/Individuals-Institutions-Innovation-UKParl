import time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import lda

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")

#=======test parameters=========
sample_size = 10000
topicnum = 100
n_iter = 100        #full run will be 8000

texts = []
with open(output_path / 'corpus.txt', encoding='utf-8') as f:
    for line in f:
        texts.append(line)
        if len(texts) == sample_size:
            break

print("documents in test:", len(texts))

CVzer = CountVectorizer(token_pattern=r"(?u)\S+", max_features=None, lowercase=True)
doc_vcnts = CVzer.fit_transform(texts)

print("vocabulary in test:", len(CVzer.get_feature_names_out()))
print("all-zero rows:", (doc_vcnts.sum(axis=1) == 0).sum())   #must be 0

#=======time the fit=========
start = time.time()
lda_model = lda.LDA(n_topics=topicnum, n_iter=n_iter, refresh=20, random_state=42)
lda_model.fit(doc_vcnts)
elapsed = time.time() - start

print("elapsed seconds:", round(elapsed, 1))

#=======count the full corpus for extrapolation=========
full_tokens = 0
full_docs = 0
with open(output_path / 'corpus.txt', encoding='utf-8') as f:
    for line in f:
        full_tokens += len(line.split())
        full_docs += 1

test_tokens = doc_vcnts.sum()

print("full docs:", full_docs)
print("full tokens:", full_tokens)

scaling = (full_tokens / test_tokens) * (8000 / n_iter)
print("estimated full run hours:", round(elapsed * scaling / 3600, 1))