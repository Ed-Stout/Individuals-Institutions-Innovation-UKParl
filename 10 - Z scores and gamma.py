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

#=========z-scoring
#raw novelty grows with window size, so a slope at scale 60 wouldn't mean the
#same as a slope at 7500. Z-scoring puts both in standard deviations from
#their own mean, which makes the slopes comparable across scales.

def z_score(values):
    return (values - values.mean()) / values.std() #NaN skipped automatically

for scale in scales:
    ntr["z_novelty_" + str(scale)] = z_score(ntr["novelty_" + str(scale)])
    ntr["z_resonance_" + str(scale)] = z_score(ntr["resonance_" + str(scale)])

print(ntr[["z_novelty_60", "z_resonance_60"]].describe().round(3))

#=========gamma aka novelty
#Barron SI Eq. 3 - resonance regressed on novelty, both z-scored.

gamma_results = []

for scale in scales:
    model_data = ntr[["z_novelty_" + str(scale), "z_resonance_" + str(scale)]].dropna()
    model_data.columns = ["z_novelty", "z_resonance"] #fixed names so one formula works for all scales

    model = smf.ols("z_resonance ~ z_novelty", data=model_data).fit()

    confidence = model.conf_int(alpha=alpha)

    gamma = model.params["z_novelty"]
    gamma_low = confidence.loc["z_novelty", 0]
    gamma_high = confidence.loc["z_novelty", 1]

    gamma_results.append({
        "scale": scale,
        "n": int(model.nobs),
        "gamma": gamma,
        "gamma_low": gamma_low,
        "gamma_high": gamma_high,
        "intercept": model.params["Intercept"],
        "r_squared": model.rsquared,
    })

    print("scale", scale,
          "| gamma", round(gamma, 4),
          "| 99% CI", round(gamma_low, 4), "to", round(gamma_high, 4),
          "| R2", round(model.rsquared, 4))

gamma_table = pd.DataFrame(gamma_results)
gamma_table.to_csv(gamma_csv, index=False)

print("saved:", gamma_csv)