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

#==========vocabulary cap============
kept_tokens = post_length              #pre-cap
kept_rows = list(range(len(post_length)))#positions in speeches
pass_number = 0

while True:
    pass_number += 1

    counts = Counter()
    for tokens in kept_tokens:
        counts.update(tokens)

    common_words = set()
    for word, n in counts.most_common(vocab_size):
        common_words.add(word)

    capped = []
    for tokens in kept_tokens:
        keep_words = []
        for word in tokens:
            if word in common_words:
                keep_words.append(word)
        capped.append(keep_words)

    #survivors carry their pre-cap tokens forward, so the next pass recounts from scratch
    survivor_tokens = []
    survivor_rows = []
    for position in range(len(capped)):
        if len(capped[position]) >= min_doc_tokens:
            survivor_tokens.append(kept_tokens[position])
            survivor_rows.append(kept_rows[position])

    removed = len(kept_tokens) - len(survivor_tokens)
    print("pass", pass_number," vocabulary", len(common_words), " removed", removed," remaining", len(survivor_tokens))

    kept_tokens = survivor_tokens
    kept_rows = survivor_rows

    if removed == 0:      #nothing left to remove, so the vocabulary is now stable
        break

post_vocab = capped       #aligned with kept_rows, since the final pass removed nothing

#==========tokens before and after=====
pre_token_cnt = 0
for tokens in input_tokens:
    pre_token_cnt += len(tokens)

post_token_cnt = 0
for tokens in post_vocab:
    post_token_cnt += len(tokens)

print("Passes needed:", pass_number)
print("Unique tokens available:", len(counts))
print("Vocabulary kept:", len(common_words))
print("Tokens before:", pre_token_cnt)
print("Tokens after:", post_token_cnt)
print("Percentage retained:", round(post_token_cnt / pre_token_cnt * 100, 1), "%")

cutoff_word, cutoff_count = counts.most_common(vocab_size)[-1]
print("Least frequent word kept:", cutoff_word, "at", cutoff_count, "uses")

speeches = speeches.reset_index(drop=True)   #kept_rows is safe

excluded = speeches.drop(index=kept_rows)
excluded.to_csv(output_path / 'excluded_short_speeches.csv', index=False, encoding='utf-8')

speeches = speeches.iloc[kept_rows].reset_index(drop=True)
speeches['tokens_vocab'] = [' '.join(tokens) for tokens in post_vocab]

doc_lengths = []
for tokens in post_vocab:
    doc_lengths.append(len(tokens))
speeches['doc_length'] = doc_lengths

#==========position after filtering ========
analysis_order = []
for position in range(len(speeches)):
    analysis_order.append(position)
speeches['analysis_order'] = analysis_order #order after filtering - key for analysis

print("Speeches removed as too short:", len(excluded)) #removed speeches
print("Speeches remaining:", len(speeches))        #total speeches
print("speech_order still sorted:", speeches['speech_order'].is_monotonic_increasing) #order

speeches.to_csv(output_path / 'Hansard_2015-20_final_corpus.csv', index=False, encoding='utf-8')