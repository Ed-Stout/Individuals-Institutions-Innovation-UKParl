from collections import Counter
import pandas as pd
import spacy

speeches = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_tokenised.csv")

pre_stops = [str(t).split() for t in speeches['tokens'].fillna('')] #adds empty string to empty cells

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
    
    cnt_table = pd.DataFrame(counts.most_common(250), columns=['token', 'count'])

    #print("Appears in most docs")
    #for word, n in doc_freq.most_common(75):
    #    print(word, round(n / len(token_list) * 100, 1), '%') #shows as %
    
    doc_pcts = []
    for token in cnt_table['token']:
        pct = round(doc_freq[token] / len(token_list) * 100, 1)
        doc_pcts.append(pct)

    cnt_table['doc_pct'] = doc_pcts
    cnt_table.to_csv(output_csv, index=False, encoding='utf-8')

    print("Sample: ", cnt_table.sample(20))
    print("Head: ", cnt_table.head(20))
    print("Counts saved in: ", output_csv)
        
print("Word counts before stopword removal")
word_cnt(pre_stops, 'pre_word_stop_frequencies.csv')

#=======Stop word removal=======
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #parser and ner 
spacy_stops = set(nlp.Defaults.stop_words)

post_stops = []
for tokens in pre_stops:
    removed = [word for word in tokens if word not in spacy_stops]
    post_stops.append(removed)

print("Word counts after stopword removal")
word_cnt(post_stops, 'post_word_stop_frequencies.csv')


