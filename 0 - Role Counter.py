import pandas as pd

#=======parameters to change=========
speeches_csv = r"G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step4.csv"
output_csv = r"G:\My Drive\Birkbeck\Project\Hansard\corpus_roles.csv"

role_columns = ["gov_role_name", "opp_role_name"] #the two columns the Cabinet split works on

pd.set_option("display.max_rows", None) #otherwise pandas truncates the middle

#==========load============
#step 4 rather than step 3, so Procedural and Division speeches are already gone -
#the counts then describe the corpus that actually reaches the model
speeches = pd.read_csv(speeches_csv, usecols=role_columns)

print(len(speeches), "speeches")

#==========count speeches per title============
#script 2 joins simultaneous posts with "; ", so split before counting, otherwise
#"Secretary of State for X; Minister for Women" reads as one unique title.
#A speech held under two posts is counted once under each, so the totals here
#deliberately exceed the speech count.
counts_table = []

for column in role_columns:
    titles = speeches[column].dropna().str.split("; ").explode()
    counts = titles.value_counts()

    print("")
    print(column, "-", len(counts), "distinct titles in the corpus")
    print(counts.to_string())

    for title in counts.index:
        counts_table.append({"column": column, "role": title, "speeches": int(counts[title])})

#==========save============
pd.DataFrame(counts_table).to_csv(output_csv, index=False, encoding="utf-8")

print("")
print("saved:", output_csv)