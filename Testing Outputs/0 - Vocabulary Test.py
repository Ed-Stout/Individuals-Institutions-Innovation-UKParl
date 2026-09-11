import pandas as pd
from collections import Counter

speeches = pd.read_csv(r'G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step6.csv')

#=======parameters to change=========
min_length = 3
caps_to_test = [6000, 7500, 10000, 12500]
min_doc_tokens = 4
retain_words = {'eu', 'uk', 'un'}

input_tokens = [str(speech).split() for speech in speeches['tokens_clean'].fillna('')]

#==========character filter============
post_length = []
for tokens in input_tokens:
    keep_words = []
    for word in tokens:
        if len(word) >= min_length or word in retain_words:
            keep_words.append(word)
    post_length.append(keep_words)

survivors = []
for tokens in post_length:
    if len(tokens) >= min_doc_tokens:
        survivors.append(tokens)

counts = Counter()
for tokens in survivors:
    counts.update(tokens)

pre_token_cnt = 0
for tokens in survivors:
    pre_token_cnt += len(tokens)

print("Speeches before:", len(post_length))
print("Speeches after short removal:", len(survivors))
print("Unique tokens available:", len(counts))
print("Tokens before cap:", pre_token_cnt)

#==========test each cap============
for vocab_size in caps_to_test:
    common_words = set()
    for word, n in counts.most_common(vocab_size):
        common_words.add(word)

    post_token_cnt = 0
    empty_speeches = 0
    for tokens in survivors:
        kept = 0
        for word in tokens:
            if word in common_words:
                kept += 1
        post_token_cnt += kept
        if kept == 0:
            empty_speeches += 1

    cutoff_word, cutoff_count = counts.most_common(vocab_size)[-1]
    print("Cap:", vocab_size, "Retained:", round(post_token_cnt / pre_token_cnt * 100, 1), "%", "Cut-off word:", cutoff_word, "at", cutoff_count, "uses", "Emptied speeches:", empty_speeches)