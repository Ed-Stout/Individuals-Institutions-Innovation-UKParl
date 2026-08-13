import pandas as pd

gov = pd.read_json(r"G:\My Drive\Birkbeck\Project\Hansard\parliamentary_roles.json")

gov = gov.explode("parliamentary_posts")
gov["role_name"] = gov["parliamentary_posts"].str["parl_post_name"]
gov["date"] = gov["date"].astype(str).str[:10]

counts = gov["role_name"].value_counts()

print(len(counts), "distinct roles\n")
print(counts.to_string())