#13 - Party size - used for analysis into how party affects NTR
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

scales = [1, 60, 250, 1000, 7500]
alpha = 0.01 #99% confidence intervals, as Barron
baseline_party = "Conservative" #differences read against the party of government throughout the period

#for lookup
party_groups = {"Conservative": "Conservative","Labour": "Labour","Labour (Co-op)": "Labour","Scottish National Party": "SNP","Liberal Democrat": "Liberal Democrat","Democratic Unionist Party": "DUP"}
residual_group = "Smaller parties"

party_sizes = {"Conservative": "Large", "Labour": "Large","SNP": "Medium","Liberal Democrat": "Medium","DUP": "Medium","Smaller parties": "Small",}

party_colours = {"Conservative": "deepskyblue", "Labour": "red", "Liberal Democrat": "orange", "SNP": "gold", "DUP": "purple", "Smaller parties": "grey"}

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
party_csv = os.path.join(source, "ntr_by_party.csv")
gamma_csv = os.path.join(source, "party_gamma_by_scale.csv")
missing_party_csv = os.path.join(source, "speeches_missing_party.csv")
figure_png = os.path.join(source, "ntr_by_party.png")

#=========load and merge
ntr = pd.read_csv(ntr_csv)

parties = pd.read_csv(corpus_csv, usecols=["analysis_order", "party"])
parties = parties.rename(columns={"analysis_order": "original_analysis_order"})

merged = ntr.merge(parties, on="original_analysis_order", how="left")

print("merged:", merged.shape)

if len(merged) != len(ntr):
    raise SystemExit("merge changed the row count - duplicate keys?")

#=================z-scoring=============== mean over std
def z_score(values):
    return (values - values.mean()) / values.std() 

for scale in scales:
    merged["z_novelty_" + str(scale)] = z_score(merged["novelty_" + str(scale)])
    merged["z_transience_" + str(scale)] = z_score(merged["transience_" + str(scale)])
    merged["z_resonance_" + str(scale)] = z_score(merged["resonance_" + str(scale)])

missing_rows = merged[merged["party"].isna()]
missing_rows.to_csv(missing_party_csv, index=False)

print("speeches with no party:", len(missing_rows), "written to", missing_party_csv) #to check previous issue with missing party names

merged = merged[merged["party"].notna()].copy() #SettingWithCopyWarning before

merged["party_group"] = merged["party"].map(party_groups).fillna(residual_group) #any leftovers go into smaller parties
merged["party_size"] = merged["party_group"].map(party_sizes) #add size of party

if merged["party_size"].isna().any():
    raise SystemExit("a party_group has no size band - check party_sizes") #check

print("Party values folded into the residual group: ", merged.loc[merged["party_group"] == residual_group, "party"].value_counts())

print(merged["party_group"].value_counts())
print(merged["party_size"].value_counts())

#========================= mean NTR per party===============
party_list = sorted(set(merged["party_group"]))
party_results = []

for scale in scales:
    print("Scale: ", scale)

    for group in party_list:
        frame = merged[merged["party_group"] == group]

        novelty = frame["z_novelty_" + str(scale)]
        transience = frame["z_transience_" + str(scale)]
        resonance = frame["z_resonance_" + str(scale)]

        party_results.append({"scale": scale, "party_group": group, "party_size": party_sizes[group],"number:": int(novelty.notna().sum()), "novelty": novelty.mean(), 
                              "transience": transience.mean(),"resonance": resonance.mean(), "novelty_sd": novelty.std(), "resonance_sd": resonance.std()})

        print(group, "n", int(novelty.notna().sum()), "novelty: ", round(novelty.mean(), 4), "transience: ", round(transience.mean(), 4), "resonance: ", round(resonance.mean(), 4))

party_table = pd.DataFrame(party_results)
party_table.to_csv(party_csv, index=False)

print("saved: ", party_csv)
print("rows: ", len(party_table))

#========================= gamma per party===============
gamma_results = []

for scale in scales:
    model_data = merged[["z_novelty_" + str(scale), "z_resonance_" + str(scale), "party_group"]].dropna()
    model_data.columns = ["z_novelty", "z_resonance", "party_group"]

    formula = "z_resonance ~ z_novelty * C(party_group, Treatment(reference='" + baseline_party + "'))" #each difference in Gamma has its own interval
    model = smf.ols(formula, data=model_data).fit()
    confidence = model.conf_int(alpha=alpha)

    baseline_gamma = model.params["z_novelty"]

    print("Scale: ", scale, "baseline: ", baseline_party, round(baseline_gamma, 4), "R2: ", round(model.rsquared, 4))

    for group in party_list:
        if group == baseline_party:
            difference = 0.0
            difference_low = 0.0
            difference_high = 0.0
        else:
            term = "z_novelty:C(party_group, Treatment(reference='" + baseline_party + "'))[T." + group + "]" #how statsmodels names the party difference
            difference = model.params[term]
            difference_low = confidence.loc[term, 0]
            difference_high = confidence.loc[term, 1]

        gamma_results.append({"scale": scale, "party_group": group, "number:": int((model_data["party_group"] == group).sum()), "gamma": baseline_gamma + difference,
                              "difference": difference, "difference_low": difference_low, "difference_high": difference_high})

        print(group, "gamma: ", round(baseline_gamma + difference, 4), "difference: ", round(difference, 4), "low: ", round(difference_low, 4), "high: ", round(difference_high, 4))

gamma_table = pd.DataFrame(gamma_results)
gamma_table.to_csv(gamma_csv, index=False)

print("saved: ", gamma_csv)
print("rows: ", len(gamma_table))

#=========plot===============
fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
measures = ["novelty", "transience", "resonance"]
positions = [(0, 0), (0, 1), (1, 0)] #row, col - gamma goes in (1, 1)

for index in range(3): #plot for each party
    measure = measures[index]
    row, column = positions[index]
    axis = axes[row, column]
    for group in party_list:
        group_table = party_table[party_table["party_group"] == group]
        axis.plot(group_table["scale"], group_table[measure], marker="o", color=party_colours[group], label=group)

    axis.axhline(0, color="grey", linewidth=0.8, linestyle="--") #corpus average
    axis.set_xscale("log")
    axis.set_xticks(scales)
    axis.set_xticklabels([str(scale) for scale in scales])
    axis.set_xlabel("Scale (speeches)")
    axis.set_title(measure.capitalize())

axis = axes[1, 1] #gamma
for group in party_list:
    group_table = gamma_table[gamma_table["party_group"] == group]
    axis.plot(group_table["scale"], group_table["gamma"], marker="o", color=party_colours[group], label=group)

axis.set_xscale("log")
axis.set_xticks(scales)
axis.set_xticklabels([str(scale) for scale in scales])
axis.set_xlabel("Scale (speeches)")
axis.set_title("Novelty effectiveness")

axes[0, 0].set_ylabel("Mean, z-scored against the corpus")
axes[1, 0].set_ylabel("Mean, z-scored against the corpus")
axes[1, 1].set_ylabel("Gamma (novelty effectiveness)")
axes[1, 1].legend(fontsize=8)
fig.savefig(figure_png, dpi=300)
plt.close(fig)
print("saved:", figure_png)