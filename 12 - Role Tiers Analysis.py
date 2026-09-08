import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf

scales = [1, 60, 250, 1000, 7500]

alpha = 0.01 #99% CIs, as Barron
reference_tier = "backbencher" #baseline all other tiers are measured against

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
role_csv = os.path.join(source, "gamma_by_role.csv")

#=========load and merge
ntr = pd.read_csv(ntr_csv)

roles = pd.read_csv(corpus_csv, usecols=["analysis_order", "role_tier", "party", "display_as"])
roles = roles.rename(columns={"analysis_order": "original_analysis_order"})

merged = ntr.merge(roles, on="original_analysis_order", how="left")

print("merged:", merged.shape)

if len(merged) != len(ntr):
    raise SystemExit("merge changed the row count - duplicate keys?")

if merged["role_tier"].isna().any():
    print("unmatched rows:", merged["role_tier"].isna().sum())
    raise SystemExit("some speeches did not pick up a role")

print(merged["role_tier"].value_counts())

other_tiers = sorted(set(merged["role_tier"]) - {reference_tier}) #built once, so coefficient names can be assembled rather than parsed
print("tiers in the model:", [reference_tier] + other_tiers)

#=========z-scoring===============
def z_score(values):
    return (values - values.mean()) / values.std()

for scale in scales:
    merged["z_novelty_" + str(scale)] = z_score(merged["novelty_" + str(scale)])
    merged["z_resonance_" + str(scale)] = z_score(merged["resonance_" + str(scale)])

role_results = []

for scale in scales:
    model_data = merged[["z_novelty_" + str(scale),"z_resonance_" + str(scale),"role_tier"]].dropna()
    model_data.columns = ["z_novelty", "z_resonance", "role_tier"]
    model_data["role_tier"] = pd.Categorical(model_data["role_tier"], categories=[reference_tier] + other_tiers)

    model = smf.ols("z_resonance ~ z_novelty * role_tier", data=model_data).fit()
    confidence = model.conf_int(alpha=alpha)

    print("")
    print("=== scale", scale, "| n", int(model.nobs), "| R2", round(model.rsquared, 4))

    base_gamma = model.params["z_novelty"]
    print(reference_tier, "gamma:", round(base_gamma, 4))

    for tier in other_tiers:
        name = "z_novelty:role_tier[T." + tier + "]"
        difference = model.params[name]

        role_results.append({"scale": scale, "tier": tier, "gamma": base_gamma + difference, "difference": difference, "ci_low_difference": confidence.loc[name, 0],
            "ci_high_difference": confidence.loc[name, 1], "p_value": model.pvalues[name]})

        print(tier, "gamma:", round(base_gamma + difference, 4),"difference: ", round(difference, 4), "99% CI: ", round(confidence.loc[name, 0], 4),"to", round(confidence.loc[name, 1], 4),
                "p: ", format(model.pvalues[name]))

role_table = pd.DataFrame(role_results)
role_table.to_csv(role_csv, index=False)

print("")
print("saved:", role_csv)
print("rows:", len(role_table))