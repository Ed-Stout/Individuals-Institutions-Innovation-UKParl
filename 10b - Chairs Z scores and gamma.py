#z scores (mean/std) and novelty efffectiveness (gamma) - chairs retained, face-validity check

import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf

scales = [1, 60, 250, 1000, 7500] #must match NTR

alpha = 0.01 #99% confidence intervals, per barron

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale_chairs.csv")
gamma_csv = os.path.join(source, "gamma_by_scale_chairs.csv")
z_csv = os.path.join(source, "hansard_ntr_z_scored_chairs.csv")
ntr = pd.read_csv(ntr_csv, parse_dates=["date"])

print("loaded:", ntr.shape)

#=================spread ================
for scale in scales:
    novelty = ntr["novelty_" + str(scale)].dropna() 
    resonance = ntr["resonance_" + str(scale)].dropna() #drop missing values where full corpus isn't there - 2w

    print("scale: ", scale)
    print("Novelty std: ", round(novelty.std(), 4))
    print("Resonance std: ", round(resonance.std(), 4))
    print("Ratio: ", round(resonance.std() / novelty.std(), 4))

for scale in scales: #catches if scales are changed in one script and not the other
    if "novelty_" + str(scale) not in ntr.columns:
        print(list(ntr.columns))
        raise SystemExit("no columns for scale " + str(scale) + " - rerun the NTR script")

#================z-scorin=========================
def z_score(values):
    return (values - values.mean()) / values.std() #NaN skipped automatically, mean over std

for scale in scales:
    ntr["z_novelty_" + str(scale)] = z_score(ntr["novelty_" + str(scale)])
    ntr["z_resonance_" + str(scale)] = z_score(ntr["resonance_" + str(scale)])

first_scale = str(scales[0])
print(ntr[["z_novelty_" + first_scale, "z_resonance_" + first_scale]].describe().round(3)) #cnt, mean, std, min, quartiles, max
ntr.to_csv(z_csv, index=False)
print("saved:", z_csv) #z scores saved

#=============gamma aka novelty effectiveness, Barron equation 3============
gamma_results = []

for scale in scales:
    model_data = ntr[["z_novelty_" + str(scale), "z_resonance_" + str(scale)]].dropna()
    model_data.columns = ["z_novelty", "z_resonance"] #fixed names so one formula works for all scales

    model = smf.ols("z_resonance ~ z_novelty", data=model_data).fit() #resonance explained by novelty
    print("scale: ", scale)
    print("rows in: ", len(ntr))
    print("scoreable: ", len(model_data))
    print("dropped: ", len(ntr) - len(model_data))

    confidence = model.conf_int(alpha=alpha) #small table, one row per coefficient

    gamma = model.params["z_novelty"] #slope - gamma
    gamma_low = confidence.loc["z_novelty", 0]
    gamma_high = confidence.loc["z_novelty", 1]

    gamma_results.append({"scale": scale,"n": int(model.nobs),"gamma": gamma, "gamma_low": gamma_low,"gamma_high": gamma_high,"intercept": model.params["Intercept"],
        "r_squared": model.rsquared})

    print("scale: ", scale)
    print("gamma: ", round(gamma, 4))
    print("99% CI: ", round(gamma_low, 4), "to", round(gamma_high, 4),"R2: ", round(model.rsquared, 4))

gamma_table = pd.DataFrame(gamma_results)
gamma_table.to_csv(gamma_csv, index=False)

print("saved:", gamma_csv)