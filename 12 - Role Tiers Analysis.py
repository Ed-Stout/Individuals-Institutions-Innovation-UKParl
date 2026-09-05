import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf

scales = [60, 250, 1000, 7500]

alpha = 0.01 #99% CIs, as Barron
reference_tier = "backbencher" #baseline all other tiers are measured against

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
role_csv = os.path.join(source, "gamma_by_role.csv")

#=========load and merge
#the NTR file carries no role information, so it comes back from the corpus.
#original_analysis_order is the key - analysis_order was renumbered after
#chairs were dropped, so it no longer matches the corpus file.

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

#=========z-scoring===============
#done across the whole corpus, not per tier, so the tiers stay comparable.
#Z-scoring within tiers would erase exactly the differences we're testing.

def z_score(values):
    return (values - values.mean()) / values.std()

for scale in scales:
    merged["z_novelty_" + str(scale)] = z_score(merged["novelty_" + str(scale)])
    merged["z_resonance_" + str(scale)] = z_score(merged["resonance_" + str(scale)])

role_results = []

for scale in scales:
    model_data = merged[["z_novelty_" + str(scale),"z_resonance_" + str(scale),"role_tier"]].dropna()

    model_data.columns = ["z_novelty", "z_resonance", "role_tier"]

    model_data["role_tier"] = pd.Categorical(model_data["role_tier"],categories=[reference_tier] + sorted(set(model_data["role_tier"]) - {reference_tier}))

    model = smf.ols("z_resonance ~ z_novelty * role_tier", data=model_data).fit()

    confidence = model.conf_int(alpha=alpha)

    print("")
    print("=== scale", scale, "| n", int(model.nobs), "| R2", round(model.rsquared, 4))

    base_gamma = model.params["z_novelty"]
    print(reference_tier, "gamma:", round(base_gamma, 4))

    for name in model.params.index:
        if name.startswith("z_novelty:"):
            tier = name.split("T.")[1].rstrip("]")
            difference = model.params[name]

            role_results.append({"scale": scale,"tier": tier,"gamma": base_gamma + difference, "difference": difference, "ci_low": confidence.loc[name, 0], "ci_high": confidence.loc[name, 1],
                "p_value": model.pvalues[name]})

            print(tier, "gamma:", round(base_gamma + difference, 4),"| difference", round(difference, 4),"| 99% CI", round(confidence.loc[name, 0], 4), "to", round(confidence.loc[name, 1], 4),"| p", format(model.pvalues[name], ".2e"))

role_table = pd.DataFrame(role_results)
role_table.to_csv(role_csv, index=False)

print("")
print("saved:", role_csv)

base_gamma = model.params["z_novelty"]
print(reference_tier, "gamma:", round(base_gamma, 4))

for name in model.params.index:
    if name.startswith("z_novelty:"):
        tier = name.split("T.")[1].rstrip("]")
        difference = model.params[name]

        role_results.append({"scale": scale,"tier": tier,"gamma": base_gamma + difference, "difference": difference, "ci_low": confidence.loc[name, 0],
            "ci_high": confidence.loc[name, 1],"p_value": model.pvalues[name]})

        print(tier, "gamma:", round(base_gamma + difference, 4),"| difference", round(difference, 4), "| 99% CI", round(confidence.loc[name, 0], 4),"to", round(confidence.loc[name, 1], 4),
                "| p", format(model.pvalues[name], ".2e"))

role_table = pd.DataFrame(role_results)
role_table.to_csv(role_csv, index=False)

print("")