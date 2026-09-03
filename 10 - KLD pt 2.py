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