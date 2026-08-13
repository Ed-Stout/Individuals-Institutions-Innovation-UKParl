import pandas as pd

speeches = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016.csv", dtype=str) #dtype = str so ids match
speaker_roles = pd.read_json(r"G:\My Drive\Birkbeck\Project\Hansard\parliamentary_roles.json")
government_roles = pd.read_json(r"G:\My Drive\Birkbeck\Project\Hansard\government_roles.json")
opposition_roles = pd.read_json(r"G:\My Drive\Birkbeck\Project\Hansard\opposition_roles.json")

#========speaker_roles=============
speaker_roles = speaker_roles.explode("parliamentary_posts") #everyone with multiple roles gets a new row for each role
speaker_roles["speaker_role_name"] = speaker_roles["parliamentary_posts"].str["parl_post_name"] #create dictionary on the parl_post_name

speaker_roles["mnis_id"] = speaker_roles["mnis_id"].astype(str) #make sure mnis_id is a string so it matches the speeches df
speaker_roles["date"] = speaker_roles["date"].astype(str).str[:10] #make sure date is a string so it matches the speeches df, keep only date and not time

speaker_roles = speaker_roles[["mnis_id", "date", "speaker_role_name"]]

speaker_roles = speaker_roles.groupby(["mnis_id", "date"])["speaker_role_name"].agg("; ".join).reset_index() #prevents duplicates by adding mtuliple roles into one cell seperated by comma. Reset index

#==========government_roles=============

government_roles = government_roles.explode("government_posts") #everyone with multiple roles gets a new row for each role
government_roles["gov_role_name"] = government_roles["government_posts"].str["gov_post_name"] #create dictionary on the gov_post_name

government_roles["mnis_id"] = government_roles["mnis_id"].astype(str) 
government_roles["date"] = government_roles["date"].astype(str).str[:10] 

government_roles = government_roles[["mnis_id", "date", "gov_role_name"]]

government_roles = government_roles.groupby(["mnis_id", "date"])["gov_role_name"].agg("; ".join).reset_index()

#==========opposition_roles=============

opposition_roles = opposition_roles.explode("opposition_posts") #everyone with multiple roles gets a new row for each role
opposition_roles["opp_role_name"] = opposition_roles["opposition_posts"].str["oppo_post_name"] #create dictionary on the opp_post_name

opposition_roles["mnis_id"] = opposition_roles["mnis_id"].astype(str) 
opposition_roles["date"] = opposition_roles["date"].astype(str).str[:10] 

opposition_roles = opposition_roles[["mnis_id", "date", "opp_role_name"]]

opposition_roles = opposition_roles.groupby(["mnis_id", "date"])["opp_role_name"].agg("; ".join).reset_index()

#==========join_all==============

speeches_join = speeches.merge(speaker_roles, on=["mnis_id", "date"], how="left") #join both
speeches_join = speeches_join.merge(government_roles, on=["mnis_id", "date"], how="left") #join both
speeches_join = speeches_join.merge(opposition_roles, on=["mnis_id", "date"], how="left") #join both

speeches_join["role"] = (speeches_join["speaker_role_name"].combine_first(speeches_join["gov_role_name"]).combine_first(speeches_join["opp_role_name"])) #all in one column

speeches_join.to_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2016-updated.csv", index=False)

#print(speaker_roles.sample(5))
#print(speeches_join.head())
#print(speaker_roles.tail())
#print(speaker_roles["role_name"].value_counts())

print(len(speeches), "speeches before join")
print(len(speeches_join), "rows after join")
print(speeches_join["role"].notna().sum(), "speeches with a role")