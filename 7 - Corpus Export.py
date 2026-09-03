import pandas as pd
from pathlib import Path

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")

use_columns = ['id', 'analysis_order', 'speech_order', 'tokens_vocab'] #reduce load
speeches = pd.read_csv(output_path / 'Hansard_2015-20_final_corpus.csv', usecols=use_columns)

speeches = speeches.sort_values('analysis_order').reset_index(drop=True) #very important for analysis

speeches['tokens_vocab'] = speeches['tokens_vocab'].fillna('') #deals with nulls

print("speeches:", len(speeches))
print("sorted:", speeches['analysis_order'].is_monotonic_increasing)

print("blank token strings:", (speeches['tokens_vocab'].fillna('').str.strip() == '').sum())
print("newlines inside tokens:", speeches['tokens_vocab'].fillna('').str.contains('\n').sum())

corpus_txt = output_path / 'corpus.txt'

with open(corpus_txt, 'w', encoding='utf-8') as text:
    for tokens in speeches['tokens_vocab']:
        text.write(tokens.strip() + '\n') #new line after each one

#count issues
line_cnt = 0
blank_lines = 0
with open(corpus_txt, encoding='utf-8') as f:
    for line in f:
        line_cnt += 1
        if line.strip() == '':
            blank_lines += 1

print("lines written:", line_cnt, "should equal", len(speeches))
print("blank lines:", blank_lines)
