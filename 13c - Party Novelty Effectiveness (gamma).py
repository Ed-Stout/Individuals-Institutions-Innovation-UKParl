#13c - Party Novelty Effectiveness
import pandas as pd
import os
import statsmodels.formula.api as smf

#==========parameters============
scales = [1, 60, 250, 1000, 7500]
alpha = 0.01 #99% confidence intervals, per barron
baseline_party = "Conservative" #differences read against the party of government

#same grouping as script 13, so the two tables can sit side by side
party_groups = {"Conservative": "Conservative", "Labour": "Labour", "Labour (Co-op)": "Labour",
                "Scottish National Party": "SNP", "Liberal Democrat": "Liberal Democrat", "Democratic Unionist Party": "DUP"}
residual_group = "Smaller parties"

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
gamma_csv = os.path.join(source, "party_gamma_by_scale.csv")

#==========load and merge============
usecols = ["original_analysis_order"]
for scale in scales:
    usecols.append("novelty_" + str(scale))
    usecols.append("resonance_" + str(scale))

ntr = pd.read_csv(ntr_csv, usecols=usecols) #transience is not used for a slope
parties = pd.read_csv(corpus_csv, usecols=["analysis_order", "party"])
parties = parties.rename(columns={"analysis_order": "original_analysis_order"})
merged = ntr.merge(parties, on="original_analysis_order", how="left")

if len(merged) != len(ntr):
    raise SystemExit("merge changed the row count - duplicate keys?")

print("speeches with no party, dropped:", int(merged["party"].isna().sum()))
merged = merged[merged["party"].notna()]

merged["party_group"] = merged["party"].map(party_groups).fillna(residual_group)
print(merged["party_group"].value_counts())

other_groups = sorted(set(merged["party_group"]) - {baseline_party}) #built once, so term names can be assembled rather than parsed

#==========z-score============
def z_score(values):
    return (values - values.mean()) / values.std()

for scale in scales:
    merged["z_novelty_" + str(scale)] = z_score(merged["novelty_" + str(scale)])
    merged["z_resonance_" + str(scale)] = z_score(merged["resonance_" + str(scale)])
    merged = merged.drop(columns=["novelty_" + str(scale), "resonance_" + str(scale)])

#==========gamma by party============
gamma_results = []

for scale in scales:
    columns = ["z_novelty_" + str(scale), "z_resonance_" + str(scale), "party_group"]
    model_data = merged[columns].dropna()
    model_data.columns = ["z_novelty", "z_resonance", "party_group"]

    formula = "z_resonance ~ z_novelty * C(party_group, Treatment(reference='" + baseline_party + "'))"
    model = smf.ols(formula, data=model_data).fit()
    confidence = model.conf_int(alpha=alpha)

    baseline_gamma = model.params["z_novelty"]

    gamma_results.append({"scale": scale, "party_group": baseline_party,
                          "n": int((model_data["party_group"] == baseline_party).sum()),
                          "gamma": baseline_gamma, "difference": 0.0,
                          "difference_low": 0.0, "difference_high": 0.0})

    for group in other_groups:
        term = "z_novelty:C(party_group, Treatment(reference='" + baseline_party + "'))[T." + group + "]"

        gamma_results.append({"scale": scale, "party_group": group,"n": int((model_data["party_group"] == group).sum()),"gamma": baseline_gamma + model.params[term],
                              "difference": model.params[term], "difference_low": confidence.loc[term, 0], "difference_high": confidence.loc[term, 1]})

    print("Scale: ", scale, "baseline: ", baseline_party, round(baseline_gamma, 4), "R2: ", round(model.rsquared, 4))

gamma_table = pd.DataFrame(gamma_results)
gamma_table.to_csv(gamma_csv, index=False)

print(gamma_table.round(4).to_string(index=False))
print("saved:", gamma_csv)