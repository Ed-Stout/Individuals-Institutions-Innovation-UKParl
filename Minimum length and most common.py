import pandas as pd 
from pathlib import Path

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")
speeches = pd.read_csv(output_path / '')

input_tokens = [str(t).split() for t in speeches['tokens_clean'].fillna('')]

#==========minimu tokens of 3======
min_length = 3 # see impact of different lengths
retain_words = {'eu', 'uk', 'un'} # words from bigrams/trigrams / most common??

post_length = []
for tokens in tokens_in:
    keep_words = []
    for word in tokens:
        if len(word) >= min_length or word in retain_words:
            keep_words.append(word)
    post_length.append(keep_words)

