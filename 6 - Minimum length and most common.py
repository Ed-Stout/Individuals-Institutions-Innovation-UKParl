import pandas as pd 
from pathlib import Path
from collections import Counter

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
speeches = pd.read_csv(output_path / 'Hansard_Dataset_cleaned.csv')

input_tokens = [str(speech).split() for speech in speeches['tokens_clean'].fillna('')]

#==========minimu tokens of 3======
min_length = 3 # see impact of different lengths
retain_words = {'eu', 'uk', 'un'} # words from bigrams/trigrams / most common??

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
for word, n in counts.most_common(10000): #key variable to change
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

cutoff_word, cutoff_count = counts.most_common(10000)[-1]
print("Least frequent word kept:", cutoff_word, "at", cutoff_count, "uses")

#==========save============
speeches['tokens_vocab'] = [' '.join(word) for word in post_vocab]
speeches.to_csv(output_path / 'Hansard_Dataset_vocab.csv', index=False, encoding='utf-8')