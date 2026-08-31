import pandas as pd

gov = pd.read_json(r"G:\My Drive\Birkbeck\Project\Hansard\government_roles.json")

gov = gov.explode("government_posts")
gov["role_name"] = gov["government_posts"].str["gov_post_name"]
gov["date"] = gov["date"].astype(str).str[:10]

counts = gov["role_name"].value_counts().head(30)

print(len(counts), "distinct roles\n")
print(counts.to_string())