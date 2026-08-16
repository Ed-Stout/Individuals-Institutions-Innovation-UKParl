from collections import Counter
import pandas as pd
import spacy

speeches = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_tokenised.csv")

pre_stops = [str(t).split() for t in speeches['tokens'].fillna('')] #adds empty string to empty cells

#==========Count words==============
def word_cnt(token_list):
    all_words = ' '.join(speeches['tokens']).split()
    counts = Counter(all_words)

    print("total tokens: ", counts.values())
    print("Unique tokens: ", len(counts))

    print("Most common words")
    for word, n in counts.most_common(150): #most common words
        print(word, n)

    doc_freq = Counter() #document frequency
    for tokens in token_list:
        doc_freq.update(set(tokens.split())) #changes to set

    print("Appears in most docs")
    for word, n in doc_freq.most_common(75):
        print(word, round(n / len(list) * 100, 1), '%') #shows as %
        
print("Word counts before stopword removal")
word_cnt(pre_stops)

#=======Stop word removal=======
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #parser and ner 
spacy_stops = set(nlp.Defaults.stop_words)

post_stops = []
for tokens in pre_stops:
    removed = [word for word in tokens if word not in spacy_stops]
    post_stops.append(removed)

print("Word counts after stopword removal")
word_cnt(post_stops)
