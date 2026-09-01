import pandas as pd
from pathlib import Path

output_path = Path(r"G:\My Drive\Birkbeck\Project\Hansard")

corpus = pd.read_csv(output_path / 'Hansard_2015-20_final_corpus.csv')

add_columns = ['id', 'party', 'person_id', 'constituency', 'role', 'major_heading', 'minor_heading', 'speech']

prev_corpus = pd.read_csv(output_path / 'hansard_speeches_2015-20_step4.csv',
                       usecols=add_columns)

rows_before = len(corpus) # to check later
corpus = corpus.merge(prev_corpus, on='id', how='left', validate='1:1') #left join sql

print("Rows before: ", rows_before)
print("Rows after: ", len(corpus))
print("rows with empty speeches: ", corpus['speech'].isna().sum())
print("Analysis order still sorted?", corpus['analysis_order'].is_monotonic_increasing)
print(corpus['party'].value_counts().head(10))

corpus.to_csv(output_path / 'Hansard_2015-20_final_corpus.csv', index=False, encoding='utf-8')