#14 - Change Point Detection
import numpy as np
import pandas as pd
import os
#import statsmodels.formula.api as smf

#==========parameters============
scales = [1, 60, 250, 1000, 7500] #must match the NTR 
alpha = 0.01 #99% confidence 
window_days = 180 #calendar days either side of the recess
min_effective_windows = 10
min_epoch_rows = 1000

#=====================soruces==========================
source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
epoch_csv = os.path.join(source, "epoch_gamma_by_event.csv")
epoch_txt = os.path.join(source, "epoch_diagnostics.txt")
plot_png = os.path.join(source, "epoch_gamma_by_event.png")

events = pd.read_csv(events_csv).to_dict("records") #dates live in source file

usecols = ["original_analysis_order", "date"]
for scale in scales:
    usecols.append("novelty_" + str(scale))
    usecols.append("resonance_" + str(scale))

events_csv = os.path.join(source, "change_point_events.csv")
ntr = pd.read_csv(ntr_csv, usecols=usecols, parse_dates=["date"]) #usecols not a plain read, transience is never used here

print("loaded:", ntr.shape)
print("date range:", ntr["date"].min().date(), "to", ntr["date"].max().date())

#==========z-score============
def z_score(values):
    return (values - values.mean()) / values.std() #NaN skipped automatically, mean over std

event_flags = {}

    flags = [] #caveats derived from the dates, not typed into the events file

    if after_end > corpus_end:
        flags.append("after epoch truncated by the end of the corpus")

    for other in events:
        if other["name"] != event["name"]:     #a window reaching across another event date means the two cannot be separated
            other_date = pd.Timestamp(other["event_date"])
            if before_start <= other_date <= after_end:
                flags.append("window also contains " + other["name"] + " - " + other["event_date"])

    for flag in flags:
        print("FLAG:", flag)

    event_flags[event["name"]] = "; ".join(flags)
