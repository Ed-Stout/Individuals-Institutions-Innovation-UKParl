import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step6.csv"
output_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard_step6_2015-20_sample.csv"

data_for_review = pd.read_csv(input_csv)

# ======Data review======
print(data_for_review.shape)
print(list(data_for_review.columns))
#print(data_for_review[['speech', 'tokens']].head(20))

# ====== Sample for manual review ======
#columns_for_review = ['display_as', 'party', 'date', 'constituency', 'major_heading', 'minor_heading','speech', 'tokens', 'role']

#sample = data_for_review.head(5000)
#sample = data_for_review.tail(1000)
sample = data_for_review.sample(2000, random_state=42)#[columns_for_review]

print(sample.head(20))
sample.to_csv(output_csv, index=False, encoding='utf-8')