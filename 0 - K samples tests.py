#K samples tests - used to check which one of K=50, K=100, or K=150 should be completed
import time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import lda

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
save_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
save_path.mkdir(parents=True, exist_ok=True)

#=======parameters =========
#k_values = [50, 100, 150]
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
for position in range(0, len(all_texts), step): #every 7th step - variable
    texts.append(all_texts[position])
#texts = texts[:sample_size]

print("documents in sample:", len(texts)) #sample size

CVzer = CountVectorizer(token_pattern=r"(?u)\S+", max_features=None) #whitespaces, vocab already capped and lowercased
doc_vcnts = CVzer.fit_transform(texts) #word counts
vocabulary = CVzer.get_feature_names_out() #words

zero_rows = (doc_vcnts.sum(axis=1) == 0).sum() #check no zeroes

print("vocabulary:", len(vocabulary)) #check numbers of words (probs less than 10k)
print("all-zero rows:", zero_rows) 

if zero_rows > 0:
    raise SystemExit("empty documents in sample - check the corpus export") #testin

#==========fit============
start = time.time()

lda_model = lda.LDA(n_topics=topicnum, n_iter=n_iter, refresh=100, alpha=alpha, eta=eta, random_state=random_state) #parameters
lda_model.fit(doc_vcnts) #fit transform not needed

elapsed = time.time() - start
print("elapsed mins:", round(elapsed / 60, 1))

topic_word = lda_model.topic_word_ #k x vocab table

#=======exclusivity - distinctiveness======
top_n = 20 #standard practice
word_totals = topic_word.sum(axis=0) #total prob / all topics

exclusivities = []
for k in range(topicnum):
    top_indices = np.argsort(topic_word[k])[::-1][:top_n] #top 20 by position, largest first
    shares = []
    for index in top_indices:
        shares.append(topic_word[k][index] / word_totals[index]) #how common across corpus
    exclusivities.append(float(np.mean(shares))) #average distinctiveness per topic

mean_exclusivity = float(np.mean(exclusivities)) #average across all topics
print("mean exclusivity:", round(mean_exclusivity,2))

#=======near-duplicate topics=========
top_sets = []
for k in range(topicnum):
    top_indices = np.argsort(topic_word[k])[::-1][:10] #top10
    top_sets.append(set(top_indices)) #no duplicates, intersection

duplicate_pairs = 0
for a in range(topicnum):
    for b in range(a + 1, topicnum): #+1 means they don't compare to themselves
        if len(top_sets[a] & top_sets[b]) >= 5: #count intersections
            duplicate_pairs += 1

total_pairs = topicnum * (topicnum - 1) / 2
duplicate_rate = duplicate_pairs / total_pairs

print("topic pairs sharing 5+ of their top 10 words:", duplicate_pairs)
print("as a share of all pairs:", round(duplicate_rate,4)) #to be easier to compare

#=======top words for manual reading=========
with open(save_path / f'sample_topwords_k{topicnum}.txt', 'w', encoding='utf-8') as f:
    for k in range(topicnum):
        top_indices = np.argsort(topic_word[k])[::-1][:15] #top 15
        words = []
        for index in top_indices: #top_indices has column numbers and vocabulary has words
            words.append(vocabulary[index])
        word_list = ' '.join(words)
        score = round(exclusivities[k], 2)
        f.write("topic " + str(k) + " (excl " + str(score) + "): " + word_list + "\n")

np.savetxt(save_path / f'loglik_sample_k{topicnum}.txt', lda_model.loglikelihoods_)

#=======one line per K, for comparison table=========
with open(save_path / f'k_comparison_k{topicnum}.txt', 'w', encoding='utf-8') as f:
    f.write("K: " + str(topicnum) + "\n")
    f.write("sample size: " + str(len(texts)) + "\n")
    f.write("vocabulary: " + str(len(vocabulary)) + "\n")
    f.write("n_iter: " + str(n_iter) + "\n")
    f.write("mean exclusivity: " + str(round(mean_exclusivity, 4)) + "\n")
    f.write("duplicate pairs: " + str(duplicate_pairs) + "\n")
    f.write("duplicate rate: " + str(round(duplicate_rate, 4)) + "\n")
    f.write("final log likelihood: " + str(lda_model.loglikelihoods_[-1]) + "\n")
    f.write("elapsed minutes: " + str(round(elapsed / 60, 1)) + "\n")