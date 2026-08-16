from collections import Counter
import pandas as pd
import spacy

speeches = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_tokenised.csv")

all_words = ' '.join(speeches['tokens']).split()
counts = Counter(all_words)

for word, n in counts.most_common(150): #most common words
    print(word, n)

doc_freq = Counter() #document frequency
for tokens in df['tokens']:
    doc_freq.update(set(tokens.split()))

for word, n in doc_freq.most_common(50):
    print(word, round(n / len(df) * 100, 1), '%')

