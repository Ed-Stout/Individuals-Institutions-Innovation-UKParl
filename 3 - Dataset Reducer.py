import pandas as pd

df = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2015_20-step3.csv", encoding='utf-8')

print("Rows before: ", len(df))
drop_speeches = ['Procedural', 'Division'] #speeches which are routine procedures not useful for analysis

excluded = df[df['speech_class'].isin(drop_speeches)]
print("Rows removed: ", len(excluded))

df = df[~df['speech_class'].isin(drop_speeches)]

df = df[['id', 'speech', 'display_as', 'party', 'constituency',
         'mnis_id', 'person_id', 'colnum', 'date', 'speech_class',
         'major_heading', 'minor_heading', 'year',
         'id_date', 'id_colnum', 'id_seq', 'speech_order',
         'speaker_role_name', 'gov_role_name', 'opp_role_name',
         'role', 'role_tier']] #keep only columns which are useful

print(len(df), "rows after")
#print("speech_order still sorted:", df['speech_order'].is_monotonic_increasing) #checks order
#print("null speeches kept:", df['speech_class'].isna().sum())
print(df['role_tier'].value_counts())
print(df['speech_class'].value_counts(dropna=False))

df.to_csv(r'G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step4.csv', index=False, encoding='utf-8')