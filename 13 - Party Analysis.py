#13 - Prty size
import numpy as np
import pandas as pd
import os
#import matplotlib.pyplot as plt

scales = [1, 60, 250, 1000, 7500]

party_groups = {"Conservative": "Conservative","Labour": "Labour","Labour (Co-op)": "Labour","Scottish National Party": "SNP","Liberal Democrat": "Liberal Democrat""Democratic Unionist Party": "DUP",
}
residual_group = "Smaller parties"

party_sizes = {"Conservative": "Large", "Labour": "Large","SNP": "Medium","Liberal Democrat": "Medium","DUP": "Medium","Smaller parties": "Small",}

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
party_csv = os.path.join(source, "ntr_by_party.csv")
#####missing_party_csv = os.path.join(source, "speeches_missing_party.csv")
##figure_png = os.path.join(source, "ntr_by_party.png")

#=========load and merge
ntr = pd.read_csv(ntr_csv)

parties = pd.read_csv(corpus_csv, usecols=["analysis_order", "party"])
parties = parties.rename(columns={"analysis_order": "original_analysis_order"})

merged = ntr.merge(parties, on="original_analysis_order", how="left")

#print("merged:", merged.shape)

if len(merged) != len(ntr):
    raise SystemExit("merge changed the row count - duplicate keys?")