#14 - Change Point Detection
#import numpy as np
import pandas as pd
import os
#import statsmodels.formula.api as smf

#==========parameters============
scales = [1, 60, 250, 1000, 7500] #must match the NTR script
alpha = 0.01 #99% confidence intervals, as Barron
window_days = 180 #calendar days either side of the recess. the only free choice in this design
min_effective_windows = 10 #an epoch holding fewer than this many non-overlapping windows is too thin to read
min_epoch_rows = 1000 #warn below this, dates are fixed so a small epoch is real, not an error

#=====================soruces#############
source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
epoch_csv = os.path.join(source, "epoch_gamma_by_event.csv")
epoch_txt = os.path.join(source, "epoch_diagnostics.txt")
plot_png = os.path.join(source, "epoch_gamma_by_event.png")

#==========load============
usecols = ["original_analysis_order", "date"]
for scale in scales:
    usecols.append("novelty_" + str(scale))
    usecols.append("resonance_" + str(scale))

ntr = pd.read_csv(ntr_csv, usecols=usecols, parse_dates=["date"]) #usecols not a plain read, transience is never used here

print("loaded:", ntr.shape)
print("date range:", ntr["date"].min().date(), "to", ntr["date"].max().date())

#==========z-score============
def z_score(values):
    return (values - values.mean()) / values.std() #NaN skipped automatically, mean over std

