import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf

scales = [60, 250, 1000, 7500] #must match the NTR script

alpha = 0.01 #99% confidence intervals, as Barron

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
gamma_csv = os.path.join(source, "gamma_by_scale.csv")

#=============load
ntr = pd.read_csv(ntr_csv, parse_dates=["date"])

print("loaded:", ntr.shape)

#catches the two scripts drifting apart if scales are changed in one and not the other
for scale in scales:
    if "novelty_" + str(scale) not in ntr.columns:
        print(list(ntr.columns))
        raise SystemExit("no columns for scale " + str(scale) + " - rerun the NTR script")

#=========z-scoring
#raw novelty grows with window size, so a slope at scale 60 wouldn't mean the
#same as a slope at 7500. Z-scoring puts both in standard deviations from
#their own mean, which makes the slopes comparable across scales.

def z_score(values):
    return (values - values.mean()) / values.std() #NaN skipped automatically

for scale in scales:
    ntr["z_novelty_" + str(scale)] = z_score(ntr["novelty_" + str(scale)])
    ntr["z_resonance_" + str(scale)] = z_score(ntr["resonance_" + str(scale)])

print(ntr[["z_novelty_60", "z_resonance_60"]].describe().round(3))

#=========gamma aka novelty
#Barron SI Eq. 3 - resonance regressed on novelty, both z-scored.

gamma_results = []