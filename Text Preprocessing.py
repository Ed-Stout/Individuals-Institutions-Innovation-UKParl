import pandas as pd
import spacy

df = pd.read_csv(r'G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_final.csv', encoding='utf-8')

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #removing ner and parser aftter runtime issues

speeches = df['speech'].fillna('').astype(str).tolist() #fillna needed bc of issues with NULL values

tokenised_speeches = [] 

for speech in nlp.pipe(speeches, batch_size=200): #batched approach is quicker
    lemmas = [token.lemma_ for token in speech
            if not token.is_punct #remove punctuation
            and not token.is_space # remove whitespace
            and not token.like_num] #remove numbers
    tokens = [lemma.lower() for lemma in lemmas] #lowercase after lemmatising
    tokenised_speeches.append(tokens) #collect the tokens

df['tokens'] = [' '.join(t) for t in tokenised_speeches] #add to list

print(df['speech'].iloc[1][:200])
print()
print(df['tokens'].iloc[1][:200])

#Ensure batching preserves order of speeches, as this is key for analysis
#print(len(tokenised_speeches) == len(df))
#print(df['speech'].iloc[50][:80])
#print(df['tokens'].iloc[50][:80])

df.to_csv(r'G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_tokenised.csv', index=False, encoding='utf-8')