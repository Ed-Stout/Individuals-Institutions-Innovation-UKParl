#13b party size

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

scales = [1, 60, 250, 1000, 7500]
plot_scales = [60, 250, 1000] #middle three
alpha = 0.01 #99% confidence intervals, as barron

min_speeches = 500 
fold_into_labour = ["Labour (Co-op)"]

party_colours = {"Conservative": "deepskyblue", "Labour": "red", "Liberal Democrat": "orange", "Scottish National Party": "gold",
                 "Democratic Unionist Party": "purple", "Plaid Cymru": "green", "Social Democratic & Labour Party": "brown", "Independent": "black"}

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
corpus_csv = os.path.join(source, "Hansard_2015-20_final_corpus.csv")
seats_csv = os.path.join(source, "uk_parliament_seats_2010_2024.csv")
size_csv = os.path.join(source, "ntr_by_party_size.csv")
unmatched_csv = os.path.join(source, "speeches_no_seat_count.csv")
figure_png = os.path.join(source, "gamma_by_party_size.png")

#=========load and merge
ntr = pd.read_csv(ntr_csv, parse_dates=["date"])

parties = pd.read_csv(corpus_csv, usecols=["analysis_order", "party"])
parties = parties.rename(columns={"analysis_order": "original_analysis_order"})

merged = ntr.merge(parties, on="original_analysis_order", how="left")

print("merged:", merged.shape)

if len(merged) != len(ntr):
    raise SystemExit("merge changed the row count - duplicate keys?")

seats = pd.read_csv(seats_csv)
seats["start_date"] = pd.to_datetime(seats["start_date"], format="%d-%m-%Y")
seats["end_date"] = pd.to_datetime(seats["end_date"], format="%d-%m-%Y")

#=========z-scoring===============
def z_score(values):
    return (values - values.mean()) / values.std()

for scale in scales:
    merged["z_novelty_" + str(scale)] = z_score(merged["novelty_" + str(scale)])
    merged["z_transience_" + str(scale)] = z_score(merged["transience_" + str(scale)])
    merged["z_resonance_" + str(scale)] = z_score(merged["resonance_" + str(scale)])

#the raw columns are not used again once z-scored - dropping them halves the frame
for scale in scales:
    merged = merged.drop(columns=["novelty_" + str(scale), "transience_" + str(scale), "resonance_" + str(scale)])

print("after dropping raw columns:", merged.shape)

#=========which parliament each speech falls in===============
merged["parliament"] = ""

parliament_windows = seats[["Parliament", "start_date", "end_date"]].drop_duplicates()

for row_number in parliament_windows.index:
    window = parliament_windows.loc[row_number]
    in_window = (merged["date"] >= window["start_date"]) & (merged["date"] <= window["end_date"])
    merged.loc[in_window, "parliament"] = window["Parliament"]

print("speeches per parliament:")
print(merged["parliament"].value_counts())

speeches_in_gaps = (merged["parliament"] == "").sum()
print("speeches falling between parliaments:", speeches_in_gaps)

#=========attach the seat count===============
print("speeches with no party, dropped:", merged["party"].isna().sum())
merged = merged[merged["party"].notna()]

merged["seat_party"] = merged["party"].replace(fold_into_labour, "Labour")

#=========name check - skippable once it prints clean===============
corpus_parties = set(merged["seat_party"])
seats_parties = set(seats["Party"])

print("in the corpus, no seat row:", sorted(corpus_parties - seats_parties))
print("in the seats file, never speaks:", sorted(seats_parties - corpus_parties))
print("matched on both sides:", len(corpus_parties & seats_parties))
#=========end of name check========================================

merged = merged.merge(seats[["Parliament", "Party", "Seats"]],
    left_on=["parliament", "seat_party"], right_on=["Parliament", "Party"], how="left")

unmatched = merged[merged["Seats"].isna()]
unmatched.to_csv(unmatched_csv, index=False)

print("speeches with no seat count:", len(unmatched), "written to", unmatched_csv)
print(unmatched["party"].value_counts())

merged = merged[merged["Seats"].notna()].copy()

print("speeches with a seat count:", len(merged))

#=========mean NTR and gamma per party per parliament===============
cells = merged.groupby(["seat_party", "parliament"]).size()
cells = cells[cells >= min_speeches]

print("party-parliament cells above", min_speeches, "speeches:", len(cells))

size_results = []

for scale in scales:
    print("Scale: ", scale)

    for seat_party, parliament in cells.index:
        frame = merged[(merged["seat_party"] == seat_party) & (merged["parliament"] == parliament)]

        novelty = frame["z_novelty_" + str(scale)]
        transience = frame["z_transience_" + str(scale)]
        resonance = frame["z_resonance_" + str(scale)]

        model_data = frame[["z_novelty_" + str(scale), "z_resonance_" + str(scale)]].dropna()
        model_data.columns = ["z_novelty", "z_resonance"]

        model = smf.ols("z_resonance ~ z_novelty", data=model_data).fit()
        confidence = model.conf_int(alpha=alpha)

        gamma = model.params["z_novelty"]
        gamma_low = confidence.loc["z_novelty", 0]
        gamma_high = confidence.loc["z_novelty", 1]

        size_results.append({"scale": scale, "seat_party": seat_party, "parliament": parliament, "seats": int(frame["Seats"].iloc[0]), "n": int(novelty.notna().sum()),
            "novelty": novelty.mean(), "transience": transience.mean(), "resonance": resonance.mean(),"novelty_sd": novelty.std(), "resonance_sd": resonance.std(),
            "gamma": gamma, "gamma_low": gamma_low, "gamma_high": gamma_high})

        print(seat_party, parliament, "seats", int(frame["Seats"].iloc[0]), "n", int(novelty.notna().sum()),
            "novelty: ", round(novelty.mean(), 4), "transience: ", round(transience.mean(), 4), "resonance: ", round(resonance.mean(), 4),
            "gamma: ", round(gamma, 4), "low: ", round(gamma_low, 4), "high: ", round(gamma_high, 4))

size_table = pd.DataFrame(size_results)
size_table.to_csv(size_csv, index=False)

print("saved:", size_csv)
print("rows:", len(size_table))

#===========================================plot===============
party_list = sorted(set(size_table["seat_party"]))

fig, axes = plt.subplots(1, len(plot_scales), figsize=(5 * len(plot_scales), 4.5), constrained_layout=True)

for column in range(len(plot_scales)):
    scale = plot_scales[column]
    axis = axes[column]
    scale_table = size_table[size_table["scale"] == scale]

    for seat_party in party_list:
        party_rows = scale_table[scale_table["seat_party"] == seat_party]
        axis.errorbar(party_rows["seats"], party_rows["gamma"], yerr=[party_rows["gamma"] - party_rows["gamma_low"], party_rows["gamma_high"] - party_rows["gamma"]],
            marker="o", markersize=6, capsize=3, linestyle="none", color=party_colours.get(seat_party, "grey"), label=seat_party) #any party not listed is grey

    axis.set_xscale("log")
    axis.set_xticks([1, 3, 10, 30, 100, 300])
    axis.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    axis.set_xlabel("Seats at the general election")
    axis.set_title("Scale = " + str(scale))

axes[0].set_ylabel("Gamma (novelty effectiveness)")
axes[len(plot_scales) - 1].legend(fontsize=8)

fig.savefig(figure_png, dpi=300)
plt.close(fig)

print("saved:", figure_png)