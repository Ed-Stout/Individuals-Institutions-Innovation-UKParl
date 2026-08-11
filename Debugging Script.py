import pandas as pd

opposition_roles = pd.read_json(r"G:\My Drive\Birkbeck\Project\Hansard\opposition_roles.json")

print("Columns:", list(opposition_roles.columns))
print()

opposition_roles = opposition_roles.explode("opposition_posts")

print("First nested value:")
print(opposition_roles["opposition_posts"].iloc[0])
print()

print("A few more:")
print(opposition_roles["opposition_posts"].head(5).tolist())