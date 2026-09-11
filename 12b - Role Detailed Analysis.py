#12b - Party role - detailed
import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

scales = [1, 60, 250, 1000, 7500]

alpha = 0.01 #99% CIs, as Barron
min_speeches = 500 

governing_party = "Conservative"
official_opposition = "Labour"
fold_into_labour = ["Labour (Co-op)"]


group_colours = {"cabinet": "deepskyblue", "Con backbench": "deepskyblue", "other government": "navy", "shadow cabinet": "red", "Lab backbench": "red",
                 "other opposition": "hotpink", "minor party frontbench": "orange", "Other backbench": "orange"}#front bench and backbench share a colour

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
group_csv = os.path.join(source, "ntr_by_gov_tier.csv")
figure_png = os.path.join(source, "ntr_by_gov_tier.png")

#=========load and merge
ntr_columns = ["original_analysis_order"]
for scale in scales:
    ntr_columns.append("novelty_" + str(scale))
    ntr_columns.append("transience_" + str(scale))
    ntr_columns.append("resonance_" + str(scale))

ntr = pd.read_csv(ntr_csv, usecols=ntr_columns, engine="python")

corpus = pd.read_csv(corpus_csv, usecols=["analysis_order", "party", "gov_tier"])
corpus = corpus.rename(columns={"analysis_order": "original_analysis_order"})

merged = ntr.merge(corpus, on="original_analysis_order", how="left")

print("merged:", merged.shape)

if len(merged) != len(ntr):
    raise SystemExit("merge changed the row count - duplicate keys?")

#=========z-scoring===============
def z_score(values):
    return (values - values.mean()) / values.std()

for scale in scales:
    merged["z_novelty_" + str(scale)] = z_score(merged["novelty_" + str(scale)])
    merged["z_transience_" + str(scale)] = z_score(merged["transience_" + str(scale)])
    merged["z_resonance_" + str(scale)] = z_score(merged["resonance_" + str(scale)])

for scale in scales:
    merged = merged.drop(columns=["novelty_" + str(scale), "transience_" + str(scale), "resonance_" + str(scale)])

merged = merged[merged["party"].notna()]
merged["party"] = merged["party"].replace(fold_into_labour, "Labour")

#=========build the eight groups===============
def group_name(party, tier):
    if tier != "backbencher":
        return tier
    if party == governing_party:
        return "Con backbench"
    if party == official_opposition:
        return "Lab backbench"
    return "Other backbench"

merged["group"] = [group_name(party, tier) for party, tier in zip(merged["party"], merged["gov_tier"])]
print("party by gov_tier:")
print(pd.crosstab(merged["party"], merged["gov_tier"]))

cells = merged.groupby("group").size()
cells = cells[cells >= min_speeches]

print(merged["group"].value_counts())
print("groups above", min_speeches, "speeches:", len(cells))

group_list = sorted(cells.index)

#=========NTR and gamma per group===============
results = []

for scale in scales:
    print("=== scale", scale)

    for group in group_list:
        frame = merged[merged["group"] == group]

        novelty = frame["z_novelty_" + str(scale)]
        transience = frame["z_transience_" + str(scale)]
        resonance = frame["z_resonance_" + str(scale)]

        count = int(novelty.notna().sum())

        novelty_ci = 2.576 * novelty.std() / np.sqrt(count)
        transience_ci = 2.576 * transience.std() / np.sqrt(count)
        resonance_ci = 2.576 * resonance.std() / np.sqrt(count)

        model_data = frame[["z_novelty_" + str(scale), "z_resonance_" + str(scale)]].dropna()
        model_data.columns = ["z_novelty", "z_resonance"]

        model = smf.ols("z_resonance ~ z_novelty", data=model_data).fit()
        confidence = model.conf_int(alpha=alpha)

        gamma = model.params["z_novelty"]
        gamma_low = confidence.loc["z_novelty", 0]
        gamma_high = confidence.loc["z_novelty", 1]

        results.append({"scale": scale, "group": group, "n": count,"novelty": novelty.mean(), "novelty_ci": novelty_ci,
            "transience": transience.mean(), "transience_ci": transience_ci,"resonance": resonance.mean(), 
            "resonance_ci": resonance_ci, "gamma": gamma, "gamma_low": gamma_low, "gamma_high": gamma_high,"r_squared": model.rsquared})

        print(group, "n", count, "novelty: ", round(novelty.mean(), 4), "transience: ", round(transience.mean(), 4),"resonance: ", round(resonance.mean(), 4),
               "+/-", round(resonance_ci, 4),"novelty effectiveness: ", round(gamma, 4), "99% CI: ", round(gamma_low, 4), "to", round(gamma_high, 4))

group_table = pd.DataFrame(results)
group_table.to_csv(group_csv, index=False)

print("saved:", group_csv)
print("rows:", len(group_table))

#=========plot 1 - each measure against scale===============
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
measures = ["novelty", "transience", "resonance", "gamma"]
titles = ["Novelty", "Transience", "Resonance", "Novelty effectiveness"]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)] #row, col

for index in range(4):
    measure = measures[index]
    row, column = positions[index]
    axis = axes[row, column]

    for group in group_list:
        group_rows = group_table[group_table["group"] == group]

        if "backbench" in group:
            line_style = "--"
        else:
            line_style = "-"
        axis.plot(group_rows["scale"], group_rows[measure], marker="o", linestyle=line_style, color=group_colours.get(group, "grey"), label=group)

    axis.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    axis.set_xscale("log")
    axis.set_xticks(scales)
    axis.set_xticklabels([str(scale) for scale in scales])
    axis.set_xlabel("Scale (speeches)")
    axis.set_title(titles[index])

axes[0, 0].set_ylabel("Mean, z-scored against the corpus")
axes[1, 0].set_ylabel("Mean, z-scored against the corpus")
axes[1, 1].set_ylabel("Novelty effectiveness (gamma)")
axes[1, 1].legend(fontsize=7)

fig.savefig(figure_png, dpi=300)
plt.close(fig)

print("saved:", figure_png)

#==============plot 2 - NTR against gamma ===============
figure_two_png = os.path.join(source, "ntr_vs_gamma_by_gov_tier.png")

fig, axes = plt.subplots(1, len(scales), figsize=(4.5 * len(scales), 4.5), constrained_layout=True)

for column in range(len(scales)):
    scale = scales[column]
    axis = axes[column]

    scale_rows = group_table[group_table["scale"] == scale]

    for group in group_list:
        group_row = scale_rows[scale_rows["group"] == group]

        if "backbench" in group:
            marker_style = "s"
        else:
            marker_style = "o"

        axis.errorbar(group_row["novelty"], group_row["gamma"],
            xerr=group_row["novelty_ci"],
            yerr=[group_row["gamma"] - group_row["gamma_low"], group_row["gamma_high"] - group_row["gamma"]],
            marker=marker_style, markersize=7, capsize=3, linestyle="none", color=group_colours.get(group, "grey"), label=group)

    axis.axvline(0, color="grey", linewidth=0.8, linestyle=":") #corpus average novelty
    axis.set_xlabel("Mean novelty (z-scored)")
    axis.set_title("Scale = " + str(scale))

axes[0].set_ylabel("Novelty effectiveness (gamma)")
axes[len(scales) - 1].legend(fontsize=7)

fig.savefig(figure_two_png, dpi=300)
plt.close(fig)

print("saved:", figure_two_png)