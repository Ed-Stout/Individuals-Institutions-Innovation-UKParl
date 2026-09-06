import time
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import lda

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
save_path = Path(r"C:\Dissertation Project\LDA_output")
save_path.mkdir(parents=True, exist_ok=True)

#=======parameters =========
topicnum = 100      #Barron used 100
n_iter = 1500       #Barron used 8000
alpha = 0.1         #lda default, matches Barron 
eta = 0.01          #lda default, matches Barron 
random_state = 42   #for reproducibility

#print(datetime.now())
print("started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")) #takes ages so good to know

with open(output_path / 'corpus.txt', encoding='utf-8') as f: #one string per speech
    texts = f.readlines()

print("documents:", len(texts))

CVzer = CountVectorizer(token_pattern=r"(?u)\S+", max_features=None, lowercase=True) #text into nums, taken from Barron, no need to cap at 10000

doc_vcnts = CVzer.fit_transform(texts)
vocabulary = CVzer.get_feature_names_out()

zero_rows = (doc_vcnts.sum(axis=1) == 0).sum() #each row token cnt

if zero_rows > 0:
    raise SystemExit("empty documents in corpus.txt - LDA cannot fit them")

print("vocabulary:", len(vocabulary))
print("total tokens:", doc_vcnts.sum()) #should equal num of tokens
print("all-zero rows:", zero_rows) #should be 0

#==========fit============
start = time.time()

lda_model = lda.LDA(n_topics=topicnum, n_iter=n_iter, refresh=50, alpha=alpha, eta=eta, random_state=random_state)
#matches Barron, 

doc_topic = lda_model.fit_transform(doc_vcnts)
topic_word = lda_model.topic_word_

elapsed = time.time() - start
print("hours:", round(elapsed / 3600, 2)) #time taken

#==========checks============
print("doc_topic shape:", doc_topic.shape)
print("rows:", doc_topic.shape[0], "should equal", len(texts))
print("smallest probability:", doc_topic.min())   #must be > 0 for KLD

#==========save============
np.save(save_path / f'topic_mixtures_k{topicnum}.npy', doc_topic)
np.save(save_path / f'topics_k{topicnum}.npy', topic_word)
np.savetxt(save_path / f'loglik_full_k{topicnum}.txt', lda_model.loglikelihoods_) #change from barron to save space

with open(save_path / 'vocabulary.txt', 'w', encoding='utf-8') as f:
    for word in vocabulary:
        f.write(word + '\n')

topwords_file = save_path / f'topwords_k{topicnum}.txt'
#==========top words ===========
with open(save_path / f'topwords_k{topicnum}.txt', 'w', encoding='utf-8') as f:
    for k in range(topicnum):
        top_indices = np.argsort(topic_word[k])[::-1][:15] #fifteen ords that define each topic
        words = []
        for index in top_indices:
            words.append(vocabulary[index])
        word_list = ' '.join(words)
        f.write("topic " + str(k) + ": " + word_list + "\n")
        

#==========record parameters ========
with open(save_path / f'run_params_k{topicnum}.txt', 'w', encoding='utf-8') as f:
    f.write("finished: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("lda version: " + str(lda.__version__) + "\n")
    f.write("documents: " + str(len(texts)) + "\n")
    f.write("vocabulary: " + str(len(vocabulary)) + "\n")
    f.write("total tokens: " + str(doc_vcnts.sum()) + "\n")
    f.write("n_topics: " + str(topicnum) + "\n")
    f.write("n_iter: " + str(n_iter) + "\n")
    f.write("alpha: " + str(alpha) + "\n")
    f.write("eta: " + str(eta) + "\n")
    f.write("random_state: " + str(random_state) + "\n")
    f.write("elapsed hours: " + str(round(elapsed / 3600, 2)) + "\n")
    f.write("final log likelihood: " + str(lda_model.loglikelihoods_[-1]) + "\n")

print("saved to:", save_path)