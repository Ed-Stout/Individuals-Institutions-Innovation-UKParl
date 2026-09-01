import pandas as pd 
from pathlib import Path
from collections import Counter

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
speeches = pd.read_csv(r'G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step6.csv')

#=======parameters to change=========
min_length = 3
vocab_size = 10000
min_doc_tokens = 4
retain_words = {'eu', 'uk', 'un'} # words from bigrams/trigrams / most common??

input_tokens = [str(speech).split() for speech in speeches['tokens_clean'].fillna('')]

post_length = []
for tokens in input_tokens:
    keep_words = []
    for word in tokens:
        if len(word) >= min_length or word in retain_words:
            keep_words.append(word)
    post_length.append(keep_words)

counts = Counter()
for tokens in post_length:
    counts.update(tokens)

common_words = set() # word and count
for word, n in counts.most_common(vocab_size): #key variable to change
    common_words.add(word)

post_vocab = []
for tokens in post_length:
    keep_words = []
    for word in tokens:
        if word in common_words:
            keep_words.append(word)
    post_vocab.append(keep_words)

#==========count of tokens before and after and empties=====
pre_token_cnt = 0
for tokens in input_tokens:
    pre_token_cnt += len(tokens)

post_token_cnt = 0
for tokens in post_vocab:
    post_token_cnt += len(tokens)

empty_speeches = 0
for tokens in post_vocab:
    if len(tokens) == 0:
        empty_speeches += 1

print("Unique tokens available:", len(counts))
print("Vocabulary kept:", len(common_words))
print("Tokens before:", pre_token_cnt)
print("Tokens after:", post_token_cnt)
print("Percentage retained:", round(post_token_cnt / pre_token_cnt * 100,1), "%")
print("Speeches with zero tokens:", empty_speeches)

cutoff_word, cutoff_count = counts.most_common(vocab_size)[-1] #last word - interesting to consider how useful it is - consider addin gmore
print("Least frequent word kept:", cutoff_word, "at", cutoff_count, "uses")

#==========how short are survivors - will use this to possibly reduce more speeches ======
short_3 = 0
short_5 = 0
short_10 = 0
for tokens in post_vocab:
    if len(tokens) < 3:
        short_3 += 1
    if len(tokens) < 5:
        short_5 += 1
    if len(tokens) < 10:
        short_10 += 1

print("Speeches under 3 tokens:", short_3)
print("Speeches under 5 tokens:", short_5)
print("Speeches under 10 tokens:", short_10)

#==========remove short speeches============
speeches['tokens_vocab'] = [' '.join(tokens) for tokens in post_vocab] #into space seperated string

doc_lengths = []
for tokens in post_vocab:
    doc_lengths.append(len(tokens))

speeches['doc_length'] = doc_lengths

too_short = speeches['doc_length'] < min_doc_tokens  #remove less than min_doc_tokens

excluded = speeches[too_short]
excluded.to_csv(output_path / 'excluded_short_speeches.csv', index=False, encoding='utf-8') #save for later

speeches = speeches[~too_short].reset_index(drop=True)

#==========position after filtering ========
analysis_order = []
for position in range(len(speeches)):
    analysis_order.append(position)

speeches['analysis_order'] = analysis_order #order after filtering - key for analysis

#==========checks============
print("Speeches removed as too short:", len(excluded)) #removed speeches
print("Speeches remaining:", len(speeches))        #total speeches
print("speech_order still sorted:", speeches['speech_order'].is_monotonic_increasing) #order

#==========save============
speeches.to_csv(output_path / 'Hansard_2015-20_final_corpus.csv', index=False, encoding='utf-8')