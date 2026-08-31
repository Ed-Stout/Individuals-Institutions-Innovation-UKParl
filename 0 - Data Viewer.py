import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016_20-updated.csv"
output_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard_2016_20_sample.csv"

data_for_review = pd.read_csv(input_csv)

# ======Data review======
print(data_for_review.shape)
print(list(data_for_review.columns))
#print(data_for_review[['speech', 'tokens']].head(20))

# ====== Sample for manual review ======
#columns_for_review = ['display_as', 'party', 'date', 'constituency', 'major_heading', 'minor_heading','speech', 'tokens', 'role']

#sample = data_for_review.head(5000)
sample = data_for_review.tail(1000)
#sample = data_for_review.sample(10000)#, random_state=42)#[columns_for_review]

print(sample)
sample.to_csv(output_csv, index=False, encoding='utf-8')