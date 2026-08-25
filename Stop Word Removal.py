from collections import Counter
import pandas as pd
import spacy
from pathlib import Path
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS

speeches = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_tokenised.csv")
output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
pre_stops = [str(t).split() for t in speeches['tokens'].fillna('')] #fillna() adds empty string to empty cells

#==========Count words==============
def word_cnt(token_list, output_csv):
    counts = Counter(word for tokens in token_list for word in tokens)

    print("total tokens: ", sum(counts.values()))
    print("Unique tokens: ", len(counts))

    doc_freq = Counter() #document frequency
    for token in token_list:
        doc_freq.update(set(token)) #changes to set

    #print("Most common words")
    #for word, n in counts.most_common(250): #most common words
    #    print(word, n)
    
    cnt_table = pd.DataFrame(counts.most_common(1000), columns=['token', 'count'])

    #print("Appears in most docs")
    #for word, n in doc_freq.most_common(75):
    #    print(word, round(n / len(token_list) * 100, 1), '%') #shows as %
    
    doc_pcts = []
    for token in cnt_table['token']:
        pct = round(doc_freq[token] / len(token_list) * 100, 1)
        doc_pcts.append(pct) #percentage

    cnt_table['doc_pct'] = doc_pcts
    cnt_table.to_csv(output_path / output_csv, index=False, encoding='utf-8')

    print("Sample: ", cnt_table.sample(20))
    print("Head: ", cnt_table.head(20))
    print("Counts saved in: ", output_csv)

def phrase_detector(token_list, output_csv, min_count=20, threshold=15, use_connectors=False):

    connectors = []
    if use_connectors:
        connectors = ENGLISH_CONNECTOR_WORDS #of, the, and etc.
    else:
        connectors = frozenset() #no connectors, sets in place

    bigram_model = Phrases(token_list, min_count=min_count, threshold=threshold, connector_words=connectors) #training
    bigram = Phraser(bigram_model) #phrases

    trigram_model = Phrases(bigram[token_list], min_count=min_count, threshold=threshold, connector_words=connectors) #same again, but one more
    trigram = Phraser(trigram_model)

    add_bigrams = [bigram[tokens] for tokens in token_list] #go through bigrams in corpus
    phrased = [trigram[tokens] for tokens in add_bigrams] #add trigrams to corpus

    phrase_cnt = Counter() #count words
    for tokens in phrased:
        for word in tokens:
            if '_' in word:
                phrase_cnt[word] += 1

    print("phrases: ", len(phrase_cnt))
    for phrase, n in phrase_cnt.most_common(100):
        print(phrase, n)

    phrase_tbl = pd.DataFrame(phrase_cnt.most_common(500), columns=['phrase', 'count'])
    phrase_tbl.to_csv(output_path / output_csv, index=False, encoding='utf-8')

    return phrased

#word_cnt(pre_stops, 'pre_word_stop_frequencies.csv')

#=======Stop word removal=======
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #parser and ner 
spacy_stops = set(nlp.Defaults.stop_words)

post_spacy_stops = []
for tokens in pre_stops:
    removed = [word for word in tokens if word not in spacy_stops] #keep the word if not in spacy_stops
    post_spacy_stops.append(removed)

#word_cnt(post_spacy_stops, 'post_word_stop_frequencies.csv')

parliamentary_stopwords = {'hon', 'friend', 'gentleman', 'lady', 'member', 'house', 'speaker',
    'right', 'mr', 'mrs', 'ms', 'sir', 'dame', 'madam', 'deputy', 'chair',
    'thank', 'grateful', 'welcome', 'congratulate', 'absolutely',
    'colleague', 'bench', 'chamber', '£',}

mp_surnames = set()   #no duplicates
for name in speeches['display_as'].dropna().unique(): #skip empties, distinct
    if name.lower() == 'unknown':
        continue
    mp_surnames.add(name.strip().split()[-1].lower()) #only last names. Bc onyl last names in the chamber

phrased_a = phrase_detector(pre_stops, 'phrases_route_a.csv', use_connectors=True)

phrased_b = phrase_detector(post_spacy_stops, 'phrases_route_b.csv', use_connectors=False)