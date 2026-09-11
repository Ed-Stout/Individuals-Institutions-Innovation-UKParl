#15 - NTR and Gamma Over Time
import pandas as pd
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

#==========parameters============
scales = [1, 60, 250, 1000, 7500] #must match NTR
plot_scales = [1, 60, 250, 1000] #a month holds fewer than one window at 7500, so monthly gamma cannot be fitted there
alpha = 0.01 #99% confidence intervals, as barron
min_bin_rows = 500 #skip months too thin to fit, mostly dissolution months
smoother_span = 0.25 #fraction of months in each local fit, roughly 16 months - descriptive, not a model
pooled_gamma = {1: 0.5961, 60: 0.4961, 250: 0.3064, 1000: 0.1217, 7500: 0.0482} #from script 10, drawn as the reference line

#=====================sources==========================
source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
events_csv = os.path.join(source, "key_events.csv")
ntr_time_png = os.path.join(source, "ntr_levels_over_time.png")
gamma_time_png = os.path.join(source, "gamma_over_time.png")

events = pd.read_csv(events_csv)
event_dates = pd.to_datetime(events["event_date"], dayfirst=True) #the file is UK format, month-first parsing misreads 03/05 and 05/11

#==========load============
usecols = ["date"]
for scale in scales:
    usecols.append("novelty_" + str(scale))
    usecols.append("resonance_" + str(scale))

ntr = pd.read_csv(ntr_csv, usecols=usecols, parse_dates=["date"]) #usecols not a plain read, transience is never used here
print("loaded:", ntr.shape)
print("date range:", ntr["date"].min().date(), "to", ntr["date"].max().date())

#==========z-score============
def z_score(values):
    return (values - values.mean()) / values.std() #NaN skipped automatically, mean over std

for scale in scales:
    ntr["z_novelty_" + str(scale)] = z_score(ntr["novelty_" + str(scale)]).astype("float32")
    ntr["z_resonance_" + str(scale)] = z_score(ntr["resonance_" + str(scale)]).astype("float32")
    ntr = ntr.drop(columns=["novelty_" + str(scale), "resonance_" + str(scale)]) #raw KLD not needed again

print("after z-scoring:", ntr.shape)
print(ntr[["z_novelty_60", "z_resonance_60"]].describe().round(3))

#==========monthly levels============
ntr["month"] = ntr["date"].dt.to_period("M").dt.to_timestamp()
months = sorted(ntr["month"].unique())

level_rows = []
for month in months:
    month_mask = ntr["month"] == month
    row = {"month": month, "n": int(month_mask.sum())}

    for scale in scales:
        row["novelty_" + str(scale)] = ntr.loc[month_mask, "z_novelty_" + str(scale)].mean()
        row["resonance_" + str(scale)] = ntr.loc[month_mask, "z_resonance_" + str(scale)].mean()

    level_rows.append(row)
level_table = pd.DataFrame(level_rows)
print("months:", len(level_table), "smallest:", int(level_table["n"].min()))

#==========levels plot============
figure, axes = plt.subplots(len(scales), 1, figsize=(11, 2.4 * len(scales)), sharex=True)

for index, scale in enumerate(scales):
    axis = axes[index]

    axis.axhline(0, color="k", linewidth=0.8) #zero is the corpus mean
    for event_date in event_dates:
        axis.axvline(event_date, color="#999999", linewidth=0.8, linestyle="--")

    axis.plot(level_table["month"], level_table["novelty_" + str(scale)], color="#4C72B0", label="novelty")
    axis.plot(level_table["month"], level_table["resonance_" + str(scale)], color="#DD8452", label="resonance")

    axis.set_ylabel("scale " + str(scale))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if index == 0:
        axis.legend(fontsize=8, ncol=2)

figure.suptitle("Monthly mean novelty and resonance, z-scored corpus-wide")
figure.tight_layout()
figure.savefig(ntr_time_png, dpi=170, bbox_inches="tight")

print("saved:", ntr_time_png)

#==========monthly gamma============
gamma_rows = []

for scale in plot_scales:
    columns = ["z_novelty_" + str(scale), "z_resonance_" + str(scale)]

    for month in months:
        month_data = ntr.loc[ntr["month"] == month, columns].dropna()
        month_data.columns = ["z_novelty", "z_resonance"] 

        if len(month_data) < min_bin_rows:
            continue

        model = smf.ols("z_resonance ~ z_novelty", data=month_data).fit()
        confidence = model.conf_int(alpha=alpha)

        gamma_rows.append({"month": month, "scale": scale, "n": len(month_data),"gamma": model.params["z_novelty"],"gamma_low": confidence.loc["z_novelty", 0],"gamma_high": confidence.loc["z_novelty", 1]})

gamma_time = pd.DataFrame(gamma_rows)
print("monthly fits:", len(gamma_time))

#========= ==== =gamma trend, Barron Eq. 7============
ntr["days"] = (ntr["date"] - ntr["date"].min()).dt.days 
trend_rows = []

for scale in plot_scales:
    columns = ["z_novelty_" + str(scale), "z_resonance_" + str(scale), "days"]
    trend_data = ntr[columns].dropna()
    trend_data.columns = ["z_novelty", "z_resonance", "days"]

    model = smf.ols("z_resonance ~ z_novelty * days", data=trend_data).fit()
    confidence = model.conf_int(alpha=alpha)

    trend_rows.append({"scale": scale,"gamma_at_start": model.params["z_novelty"],"gamma_per_day": model.params["z_novelty:days"],
                       "gamma_per_day_low": confidence.loc["z_novelty:days", 0],"gamma_per_day_high": confidence.loc["z_novelty:days", 1]})

    print("scale", scale, "gamma at start", round(model.params["z_novelty"], 4), "per year", round(model.params["z_novelty:days"] * 365, 4), "99% CI", round(confidence.loc["z_novelty:days", 0] * 365, 4), "to", round(confidence.loc["z_novelty:days", 1] * 365, 4))

trend_table = pd.DataFrame(trend_rows)

#==========gamma plot============
line_colours = {1: "#0072B2", 60: "#D55E00", 250: "#009E73", 1000: "#CC79A7"} #one per scale
figure, axis = plt.subplots(figsize=(11, 6))

for event_date in event_dates:
    axis.axvline(event_date, color="#999999", linewidth=0.8, linestyle="--")

for scale in plot_scales:
    scale_rows = gamma_time[gamma_time["scale"] == scale]

    scale_days = (scale_rows["month"] - ntr["date"].min()).dt.days
    smoothed = lowess(scale_rows["gamma"], scale_days, frac=smoother_span, return_sorted=False)

    axis.plot(scale_rows["month"], smoothed, color=line_colours[scale], linewidth=2.0,label="w = " + str(scale))

axis.set_ylim(bottom=0) #zero is meaningful 
axis.set_ylabel("Gamma (novelty effectiveness)")
axis.set_xlabel("Month")
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(plot_scales),frameon=False, fontsize=10)

figure.suptitle("Monthly returns to novelty by measurement scale, local smoother", y=1.06)
figure.tight_layout()
figure.savefig(gamma_time_png, dpi=170, bbox_inches="tight")

print("saved:", gamma_time_png)