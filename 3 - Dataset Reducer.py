import pandas as pd

input_csv = r"G:\My Drive\Birkbeck\Project\Hansard\Hansard Dataset with Roles - hansard-speeches-2016-updated.csv"

df = pd.read_csv(input_csv, encoding='utf-8')

df = df[df['speech_class'] != 'Procedural']

df = df[['id', 'speech', 'display_as', 'party', 'constituency', 'mnis_id','colnum','date', 'speech_class', 'major_heading', 'minor_heading','year','speaker_role_name', 'gov_role_name', 'opp_role_name', 'role']]

df.to_csv(r'G:\My Drive\Birkbeck\Project\Hansard\Hansard_Dataset_final.csv', index=False, encoding='utf-8')