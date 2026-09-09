import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#==========parameters============
scales = [1, 60, 250, 1000, 7500]
alpha = 0.01
window_days = 180

#=====================soruces==========================
source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
epoch_csv = os.path.join(source, "epoch_gamma_by_event.csv")
plot_png = os.path.join(source, "epoch_gamma_by_event.png")
events_csv = os.path.join(source, "key_events.csv")

events = pd.read_csv(events_csv).to_dict("records")

usecols = ["original_analysis_order", "date"]
for scale in scales:
    usecols.append("novelty_" + str(scale))
    usecols.append("resonance_" + str(scale))

ntr = pd.read_csv(ntr_csv, usecols=usecols, parse_dates=["date"])

#==========z-score============
def z_score(values):
    return (values - values.mean()) / values.std()

for scale in scales:
    ntr["z_novelty_" + str(scale)] = z_score(ntr["novelty_" + str(scale)]).astype("float32")
    ntr["z_resonance_" + str(scale)] = z_score(ntr["resonance_" + str(scale)]).astype("float32")
    ntr = ntr.drop(columns=["novelty_" + str(scale), "resonance_" + str(scale)])

#==========two-epoch gamma============
results = []

for event in events:
    recess_start = pd.to_datetime(event["recess_start"], dayfirst=True)
    recess_end = pd.to_datetime(event["recess_end"], dayfirst=True)
    before_start = recess_start - pd.Timedelta(days=window_days)
    after_end = recess_end + pd.Timedelta(days=window_days)

    before_mask = (ntr["date"] >= before_start) & (ntr["date"] < recess_start)
    after_mask = (ntr["date"] >= recess_end) & (ntr["date"] <= after_end)

    for scale in scales:
        columns = ["z_novelty_" + str(scale), "z_resonance_" + str(scale)]

        before_data = ntr.loc[before_mask, columns].dropna()
        after_data = ntr.loc[after_mask, columns].dropna()
        before_data.columns = ["z_novelty", "z_resonance"]
        after_data.columns = ["z_novelty", "z_resonance"]
        before_data["after"] = 0
        after_data["after"] = 1

        model_data = pd.concat([before_data, after_data])
        model = smf.ols("z_resonance ~ z_novelty * after", data=model_data).fit()
        confidence = model.conf_int(alpha=alpha)

        gamma_before = model.params["z_novelty"]
        gamma_shift = model.params["z_novelty:after"]

        after_slope = model.t_test("z_novelty + z_novelty:after")
        after_ci = after_slope.conf_int(alpha=alpha)
        gamma_after = float(after_slope.effect[0])

        n_before = len(before_data)
        n_after = len(after_data)

        results.append({"event": event["name"], "event_date": event["event_date"], "scale": scale, "n_before": n_before, "n_after": n_after,"gamma_before": gamma_before,"gamma_before_low": confidence.loc["z_novelty", 0],
                        "gamma_before_high": confidence.loc["z_novelty", 1],"gamma_after": gamma_after,"gamma_after_low": float(after_ci[0][0]),"gamma_after_high": float(after_ci[0][1]),
                        "gamma_shift": gamma_shift,"gamma_shift_low": confidence.loc["z_novelty:after", 0],"gamma_shift_high": confidence.loc["z_novelty:after", 1],"r_squared": model.rsquared})

        print(event["name"], "scale", scale, "before", round(gamma_before, 4), "after", round(gamma_after, 4), "shift", round(gamma_shift, 4), "99% CI", round(confidence.loc["z_novelty:after", 0], 4), "to", round(confidence.loc["z_novelty:after", 1], 4), "n before", n_before, "n after", n_after)

epoch_table = pd.DataFrame(results)

#==========save============
epoch_table.to_csv(epoch_csv, index=False)
print("saved:", epoch_csv)

#==========plot============

event_names = []
for event in events:
    event_names.append(event["name"])

scale_labels = []
for scale in scales:
    scale_labels.append(str(scale))

positions = np.arange(len(scales))
figure, axes = plt.subplots(1, len(events), figsize=(3 * len(events), 4.2), sharey=True)

for index, event_name in enumerate(event_names):
    axis = axes[index]
    scale_rows = epoch_table[epoch_table["event"] == event_name].set_index("scale").loc[scales]

    error = [scale_rows["gamma_shift"] - scale_rows["gamma_shift_low"],
             scale_rows["gamma_shift_high"] - scale_rows["gamma_shift"]]

    axis.axhline(0, color="k", linewidth=1.0)
    axis.errorbar(positions, scale_rows["gamma_shift"], yerr=error, fmt="o", color="#4C72B0", markersize=6, capsize=3)

    axis.set_title(event_name, fontsize=10)
    axis.set_xticks(positions)
    axis.set_xticklabels(scale_labels)
    axis.set_xlabel("scale")
    axis.set_xlim(-0.6, len(scales) - 0.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

axes[0].set_ylabel("change in gamma\n(after minus before)")
figure.suptitle("Change in returns to novelty at five fixed events", fontsize=13)
figure.text(0.5, 0.005, "99% CI", ha="center", fontsize=9, color="#555555")
figure.tight_layout(rect=[0, 0.03, 1, 0.99])
figure.savefig(plot_png, dpi=170, bbox_inches="tight")

print("saved:", plot_png)