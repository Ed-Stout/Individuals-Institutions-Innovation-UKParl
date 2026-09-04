import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

scales = [60, 250, 1000, 7500] #must match the NTR script

grid = 150 
clip = 0.1 #percent trimmed

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
figure_png = os.path.join(source, "density_TvN_RvN.png")

ntr = pd.read_csv(ntr_csv)
print("loaded:", ntr.shape)

#=========axis limits==========================
all_novelty = []
all_transience = []
all_resonance = []

for scale in scales:
    all_novelty.append(ntr["novelty_" + str(scale)].dropna().to_numpy())
    all_transience.append(ntr["transience_" + str(scale)].dropna().to_numpy())
    all_resonance.append(ntr["resonance_" + str(scale)].dropna().to_numpy())

all_novelty = np.concatenate(all_novelty)
all_transience = np.concatenate(all_transience)
all_resonance = np.concatenate(all_resonance)

#novelty and transience share limits so the identity line sits at 45 degrees
axis_low = min(np.percentile(all_novelty, clip), np.percentile(all_transience, clip))
axis_high = max(np.percentile(all_novelty, 100 - clip), np.percentile(all_transience, 100 - clip))

#resonance is centred on zero, so the limits are symmetric
resonance_bound = np.percentile(np.abs(all_resonance), 100 - clip)

print("novelty/transience axis:", round(axis_low, 2), "to", round(axis_high, 2))
print("resonance axis:", round(-resonance_bound, 2), "to", round(resonance_bound, 2))

#=========build the histograms
#counted first, plotted second, so every panel can share one colour scale

counts_top = {}
counts_bottom = {}
fits = {}
max_count = 0

for scale in scales:
    frame = ntr[["novelty_" + str(scale),
                 "transience_" + str(scale),
                 "resonance_" + str(scale)]].dropna()

    novelty = frame["novelty_" + str(scale)].to_numpy()
    transience = frame["transience_" + str(scale)].to_numpy()
    resonance = frame["resonance_" + str(scale)].to_numpy()

    top, x_edges, y_edges_top = np.histogram2d(
        novelty, transience, bins=grid,
        range=[[axis_low, axis_high], [axis_low, axis_high]])

    bottom, x_edges, y_edges_bottom = np.histogram2d(
        novelty, resonance, bins=grid,
        range=[[axis_low, axis_high], [-resonance_bound, resonance_bound]])

    #straight line through the cloud - this is gamma in raw units
    slope, intercept = np.polyfit(novelty, resonance, 1)

    counts_top[scale] = top
    counts_bottom[scale] = bottom
    fits[scale] = (slope, intercept)

    if top.max() > max_count:
        max_count = top.max()
    if bottom.max() > max_count:
        max_count = bottom.max()

    print("scale", scale, "| points", len(novelty), "| raw slope", round(slope, 5))

print("busiest cell:", int(max_count), "speeches")

#=========plot
#log colour scale, as Barron - the centre holds thousands of speeches and the
#tails hold single figures, so a linear scale would hide the tails entirely

colour_scale = LogNorm(vmin=1, vmax=max_count)

fig, axes = plt.subplots(2, len(scales),
                         figsize=(4 * len(scales), 8),
                         sharex=True, sharey="row")

