import time
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import lda

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
save_path = Path(r"C:\Dissertation Project\LDA_output")
save_path.mkdir(parents=True, exist_ok=True)

#=======parameters to change=========
topicnum = 100      #Barron used 100
n_iter = 1500       #Barron used 8000; reduced on convergence evidence
alpha = 0.1         #lda default, matches Barron 
eta = 0.01          #lda default, matches Barron 
random_state = 42   #for reproducibility

print("started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with open(output_path / 'corpus.txt', encoding='utf-8') as f:
    texts = f.readlines()

print("documents:", len(texts))

#token_pattern changed from Barron's 
CVzer = CountVectorizer(token_pattern=r"(?u)\S+", max_features=None, lowercase=True)

doc_vcnts = CVzer.fit_transform(texts)
vocabulary = CVzer.get_feature_names_out()

zero_rows = (doc_vcnts.sum(axis=1) == 0).sum()

print("vocabulary:", len(vocabulary))
print("total tokens:", doc_vcnts.sum())
print("all-zero rows:", zero_rows)

#==========stop before fitting, not after seven hours============
if zero_rows > 0:
    raise SystemExit("all-zero rows present - do not fit")

#==========fit============
start = time.time()

lda_model = lda.LDA(n_topics=topicnum, n_iter=n_iter, refresh=50,
                    alpha=alpha, eta=eta, random_state=random_state)

doc_topic = lda_model.fit_transform(doc_vcnts)
topic_word = lda_model.topic_word_

elapsed = time.time() - start
print("elapsed hours:", round(elapsed / 3600, 2))

#==========checks============
print("doc_topic shape:", doc_topic.shape)
print("rows:", doc_topic.shape[0], "should equal", len(texts))
print("smallest probability:", doc_topic.min())   #must be > 0 for KLD

#==========save============
#np.save not np.savetxt - the text version of doc_topic is ~800MB and slow to reload
np.save(save_path / f'topic_mixtures_k{topicnum}.npy', doc_topic)
np.save(save_path / f'topics_k{topicnum}.npy', topic_word)
np.savetxt(save_path / f'loglik_full_k{topicnum}.txt', lda_model.loglikelihoods_)

with open(save_path / 'vocabulary.txt', 'w', encoding='utf-8') as f:
    for word in vocabulary:
        f.write(word + '\n')

#==========top words, for reading in the morning============
with open(save_path / f'topwords_k{topicnum}.txt', 'w', encoding='utf-8') as f:
    for k in range(topicnum):
        top_indices = np.argsort(topic_word[k])[::-1][:15]
        words = []
        for index in top_indices:
            words.append(vocabulary[index])
        f.write(f"topic {k}: " + ' '.join(words) + '\n')

