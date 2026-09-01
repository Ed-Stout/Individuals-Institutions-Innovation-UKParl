import pandas as pd 
from pathlib import Path
from collections import Counter

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
speeches = pd.read_csv(r'G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step6.csv')

#=======parameters to change=========
min_length = 3
vocab_size = 10000
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

#==========report what happened
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
print("Percentage retained:", round(post_token_cnt / pre_token_cnt * 100, 1), "%")
print("Speeches with zero tokens:", empty_speeches)

cutoff_word, cutoff_count = counts.most_common(vocab_size)[-1]
print("Least frequent word kept:", cutoff_word, "at", cutoff_count, "uses")

#==========how short are the survivors============
short_5 = 0
short_10 = 0
short_20 = 0
for tokens in post_vocab:
    if len(tokens) < 5:
        short_5 += 1
    if len(tokens) < 10:
        short_10 += 1
    if len(tokens) < 20:
        short_20 += 1

print("Speeches under 5 tokens:", short_5)
print("Speeches under 10 tokens:", short_10)
print("Speeches under 20 tokens:", short_20)

#==========remove empty speeches============
speeches['tokens_vocab'] = [' '.join(word) for word in post_vocab]

is_empty = speeches['tokens_vocab'].str.strip() == ''

excluded = speeches[is_empty]
excluded.to_csv(output_path / 'excluded_empty_speeches.csv', index=False, encoding='utf-8')

speeches = speeches[~is_empty].reset_index(drop=True)

#==========position in the analysis chain============
#speech_order is the permanent position in the full 2015-20 chain
#analysis_order is the position after filtering - the chain closes up, per Barron SI 2.4
analysis_order = []
for position in range(len(speeches)):
    analysis_order.append(position)

speeches['analysis_order'] = analysis_order

#==========checks============
print("Speeches removed as empty:", len(excluded))
print("Speeches remaining:", len(speeches))
print("speech_order still sorted:", speeches['speech_order'].is_monotonic_increasing)

#==========save============
speeches.to_csv(output_path / 'Hansard_2015-20_final_corpus.csv', index=False, encoding='utf-8')