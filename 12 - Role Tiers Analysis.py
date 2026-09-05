import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf

scales = [60, 250, 1000, 7500]

alpha = 0.01 #99% CIs, as Barron
reference_tier = "backbencher" #baseline all other tiers are measured against

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
role_csv = os.path.join(source, "gamma_by_role.csv")

#=========load and merge
#the NTR file carries no role information, so it comes back from the corpus.
#original_analysis_order is the key - analysis_order was renumbered after
#chairs were dropped, so it no longer matches the corpus file.

ntr = pd.read_csv(ntr_csv)

roles = pd.read_csv(corpus_csv, usecols=["analysis_order", "role_tier", "party", "display_as"])
roles = roles.rename(columns={"analysis_order": "original_analysis_order"})

merged = ntr.merge(roles, on="original_analysis_order", how="left")

print("merged:", merged.shape)

if len(merged) != len(ntr):
    raise SystemExit("merge changed the row count - duplicate keys?")

if merged["role_tier"].isna().any():
    print("unmatched rows:", merged["role_tier"].isna().sum())
    raise SystemExit("some speeches did not pick up a role")

print(merged["role_tier"].value_counts())