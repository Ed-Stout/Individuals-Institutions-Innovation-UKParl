import pandas as pd

speeches = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016.csv", dtype=str) #dtype = str so ids match
speaker_roles = pd.read_json(r"G:\My Drive\Birkbeck\Project\Hansard\parliamentary_roles.json")

speaker_roles = speaker_roles.explode("parliamentary_posts") #everyone with multiple roles gets a new row for each role
speaker_roles["role_name"] = speaker_roles["parliamentary_posts"].str["parl_post_name"] #create dictionary on the parl_post_name

speaker_roles["mnis_id"] = speaker_roles["mnis_id"].astype(str) #make sure mnis_id is a string so it matches the speeches df
speaker_roles["date"] = speaker_roles["date"].astype(str).str[:10] #make sure date is a string so it matches the speeches df, keep only date and not time

speaker_roles = speaker_roles[["mnis_id", "date", "role_name"]]

speeches_join = speeches.merge(speaker_roles, on=["mnis_id", "date"], how="left") #join both
speeches_join.to_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016-updated.csv", index=False)

#print(speaker_roles.sample(5))
#print(speaker_roles.head())