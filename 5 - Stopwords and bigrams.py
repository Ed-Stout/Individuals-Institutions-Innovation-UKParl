from collections import Counter
import pandas as pd
import spacy
from pathlib import Path
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS

output_path = Path(r'G:\My Drive\Birkbeck\Project\Hansard')
#output_csv = output_path / 'hansard_speeches_2015-20_step6.csv'
output_csv = r'G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step6.csv'

use_columns = ['id', 'display_as', 'date', 'speech_order', 'role_tier', 'tokens'] #took too long to run - use only essential columns
speeches = pd.read_csv(r'G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step5.csv', usecols=use_columns)

pre_stops = [str(speech).split() for speech in speeches['tokens'].fillna('')] #fillna() adds empty string to empty cells
speeches = speeches.drop(columns=['tokens']) #drop this column once loaded

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

def phrase_detector(token_list, output_csv, min_count=100, threshold=15, use_connectors=False):

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

    del add_bigrams #prevents runtime issues

    phrase_cnt = Counter() # count words
    for tokens in phrased:
        phrases = []
        for word in tokens:
            if '_' in word:
                phrases.append(word)
        phrase_cnt.update(phrases)

    print("phrases: ", len(phrase_cnt))
    for phrase, n in phrase_cnt.most_common(100):
        print(phrase, n)

    phrase_tbl = pd.DataFrame(phrase_cnt.most_common(500), columns=['phrase', 'count'])
    phrase_tbl.to_csv(output_path / output_csv, index=False, encoding='utf-8')

    return phrased

#word_cnt(pre_stops, 'pre_word_stop_frequencies.csv')

#most common unhelpful words for analysis added to manual list
parliamentary_stopwords = {'hon', 'friend', 'gentleman', 'lady', 'member', 'house', 'speaker',
    'right', 'mr', 'mrs', 'ms', 'sir', 'dame', 'madam', 'deputy', 'chair',
    'thank', 'grateful', 'welcome', 'congratulate', 'absolutely',
    'colleague', 'bench', 'chamber', '£'}

#=======Stop word removal=======
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #parser and ner 
spacy_stops = set(nlp.Defaults.stop_words)

post_spacy_stops = []
for tokens in pre_stops:
    removed = [word for word in tokens if word not in spacy_stops] #keep the word if not in spacy_stops
    post_spacy_stops.append(removed)

del pre_stops #reduce load

#========DIAGNOSTICS - before phrase detection========
diag_counts = Counter()
for tokens in post_spacy_stops:
    diag_counts.update(tokens)

print("member:", diag_counts['member'], "members:", diag_counts['members'])
print("bill:", diag_counts['bill'], "bills:", diag_counts['bills'])
print("minister:", diag_counts['minister'], "ministers:", diag_counts['ministers'])
print("gentleman:", diag_counts['gentleman'], "gentlemen:", diag_counts['gentlemen'])

mp_surnames_check = set()
for name in speeches['display_as'].dropna().unique():
    if name.lower() == 'unknown':
        continue
    mp_surnames_check.add(name.strip().split()[-1].lower())

common_check = set()
for word, n in diag_counts.most_common(7500):
    common_check.add(word)

print("Surnames that are also common words:")
print(sorted(mp_surnames_check.intersection(common_check)))

#word_cnt(post_spacy_stops, 'post_word_stop_frequencies.csv')
#includes the top procedural phrases from earlier analysis which are not useful topics for analysis

#=====compare the two phrases routes======= decide on connectors
#phrased_a = phrase_detector(pre_stops, 'phrases_route_a.csv', use_connectors=True) #com
phrased_b = phrase_detector(post_spacy_stops, 'phrases_route_b.csv', use_connectors=False)

del post_spacy_stops #prevent runtime issues

#most common unhelpful phrases for analysis added to manual list
procedural_phrases = {'point_order', 'madam_deputy_speaker', 'mr_speaker', 'dispatch_box',
    'hon_friend', 'hon_member', 'hon_gentleman', 'hon_lady', 'hon_learn',
    'right_hon_friend', 'right_hon_gentleman', 'right_hon_lady',
    'hon_friend_member', 'thank_hon_friend', 'agree_hon_friend',
    'give_way', 'secure_debate', 'answer_question'}

domain_stops = parliamentary_stopwords.union(procedural_phrases)

#==========domain stopword removal============
post_domain_stops = [] 
for tokens in phrased_b:
    words_kept = []
    for word in tokens:
        if word not in domain_stops:
            words_kept.append(word)
    post_domain_stops.append(words_kept) #loop to go through each word and remove domain stopwords

del phrased_b #prevent runtime issues

word_cnt(post_domain_stops, 'post_domain_stop_frequencies.csv') #output most common words after

mp_surnames = set()   #no duplicates
for name in speeches['display_as'].dropna().unique(): #skip empties, distinct
    if name.lower() == 'unknown':
        continue
    mp_surnames.add(name.strip().split()[-1].lower()) #only last names. Bc onyl last names in the chamber

#========SURNAMES AND COMMON WORDS CHECK=========
counts = Counter() #count common words
for tokens in post_domain_stops:
    counts.update(tokens)

common_words = set()
for word, n in counts.most_common(7500):
    common_words.add(word)
print("Surnames that are also common words:")
common_surnames = mp_surnames.intersection(common_words) #& is intersection function - any MP surname which is common word is kept
print(common_surnames)

keep_anyway = {'day', 'green', 'double', 'fox', 'slaughter', 'pound', 'law',
    'cash', 'bone', 'brake', 'glass', 'champion', 'buck', 'west',
    'main', 'parish', 'burden', 'hall', 'churchill', 'black', 'white', 'young',
    'long'}
mp_surnames = mp_surnames - keep_anyway

#==========surname stopword removal=============
post_surname_stops = []
for tokens in post_domain_stops:
    words_kept = []
    for word in tokens:
        if word not in mp_surnames:
            words_kept.append(word)
    post_surname_stops.append(words_kept)

del post_domain_stops

word_cnt(post_surname_stops, 'post_surname_stop_frequencies.csv') #output most common words after

#==========surnames surviving inside phrases============
#removal is exact-match, so 'corbyn' goes but 'corbyn_say' does not
leaked = Counter()
for tokens in post_surname_stops:
    for word in tokens:
        if '_' in word:
            for part in word.split('_'):
                if part in mp_surnames:
                    leaked[word] += 1

print("distinct phrases containing a surname: ", len(leaked))
for phrase, n in leaked.most_common(30):
    print(phrase, n)

leak_tbl = pd.DataFrame(leaked.most_common(500), columns=['phrase', 'count'])
leak_tbl.to_csv(output_path / 'surnames_inside_phrases.csv', index=False, encoding='utf-8')

#==========speeches left with nothing============
empty = 0
for tokens in post_surname_stops:
    if len(tokens) == 0:
        empty += 1

speeches['tokens_clean'] = [' '.join(speech) for speech in post_surname_stops]
speeches.to_csv(output_csv, index=False, encoding='utf-8')