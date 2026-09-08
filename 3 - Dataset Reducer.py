import pandas as pd

df = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2015_20-step3.csv", encoding='utf-8')
excluded_csv = r"G:\My Drive\Birkbeck\Project\Hansard\excluded_unattributed_speeches.csv"

print("Rows before: ", len(df))
drop_speeches = ['Procedural', 'Division'] #speeches which are routine procedures not useful for analysis

excluded = df[df['speech_class'].isin(drop_speeches)]
print("Rows removed: ", len(excluded))

df = df[~df['speech_class'].isin(drop_speeches)]

no_person = df['person_id'].isna()
print("no person_id but has mnis_id: ", (no_person & df['mnis_id'].notna()).sum()) #where both are missing

excluded_speaker = df[no_person]
print("Unattributed speeches removed: ", len(excluded_speaker)) #cnt for write up
print(excluded_speaker['display_as'].value_counts(dropna=False).head(10))

excluded_speaker.to_csv(excluded_csv, index=False, encoding='utf-8') #audit trail

df = df[~no_person].reset_index(drop=True)

df = df[['id', 'speech', 'display_as', 'party', 'constituency',
         'mnis_id', 'person_id', 'date', 'speech_class',
         'major_heading', 'minor_heading', 'year', 'id_colnum', 'id_seq', 'speech_order',
         'speaker_role_name', 'gov_role_name', 'opp_role_name',
         'role', 'role_tier','gov_tier']] #keep only columns which are useful

print(len(df), "rows after")
#print("still sorted?", df['speech_order'].is_monotonic_increasing) #checks order
#print("null speeches kept:", dfRemoval['speech_class'].isna().sum())
print(df['role_tier'].value_counts())
print(df['gov_tier'].value_counts())
print(df['speech_class'].value_counts(dropna=False))

df.to_csv(r'G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20_step4.csv', index=False, encoding='utf-8')