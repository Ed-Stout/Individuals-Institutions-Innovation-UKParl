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

axis_low = min(np.percentile(all_novelty, clip), np.percentile(all_transience, clip)) #novelty and transience share limits
axis_high = max(np.percentile(all_novelty, 100 - clip), np.percentile(all_transience, 100 - clip))

resonance_bound = np.percentile(np.abs(all_resonance), 100 - clip) #centred on zero

print("novelty/transience axis:", round(axis_low, 2), "to", round(axis_high, 2))
print("resonance axis:", round(-resonance_bound, 2), "to", round(resonance_bound, 2))

#=========build the histograms
counts_top = {}
counts_bottom = {}
fits = {}
max_count = 0

for scale in scales:
    frame = ntr[["novelty_" + str(scale),"transience_" + str(scale), "resonance_" + str(scale)]].dropna()

    novelty = frame["novelty_" + str(scale)].to_numpy()
    transience = frame["transience_" + str(scale)].to_numpy()
    resonance = frame["resonance_" + str(scale)].to_numpy() #cnt

    top, x_edges, y_edges_top = np.histogram2d(novelty, transience, bins=grid,range=[[axis_low, axis_high], [axis_low, axis_high]])

    bottom, x_edges, y_edges_bottom = np.histogram2d(novelty, resonance, bins=grid,range=[[axis_low, axis_high], [-resonance_bound, resonance_bound]])

    slope, intercept = np.polyfit(novelty, resonance, 1) #gamma raw

    counts_top[scale] = top
    counts_bottom[scale] = bottom
    fits[scale] = (slope, intercept)

    if top.max() > max_count:
        max_count = top.max()
    if bottom.max() > max_count:
        max_count = bottom.max()

    print("scale: ", scale, "points: ", len(novelty), "raw slope: ", round(slope, 5))

print("busiest cell:", int(max_count), "speeches")

#=========plot===========================
colour_scale = LogNorm(vmin=1, vmax=max_count)

fig, axes = plt.subplots(2, len(scales),figsize=(4 * len(scales), 8),sharex=True, sharey="row")

for column in range(len(scales)):
    scale = scales[column]
    top_axis = axes[0, column]
    bottom_axis = axes[1, column]

    #empty cells masked so they render as background, not as the lowest colour
    top = np.ma.masked_where(counts_top[scale] == 0, counts_top[scale])
    bottom = np.ma.masked_where(counts_bottom[scale] == 0, counts_bottom[scale])

    #.T because histogram2d returns x down the rows, pcolormesh wants x across
    mesh = top_axis.pcolormesh(x_edges, y_edges_top, top.T,norm=colour_scale, cmap="viridis")

    bottom_axis.pcolormesh(x_edges, y_edges_bottom, bottom.T, norm=colour_scale, cmap="viridis")

    #identity line - speeches on it are equally surprising to past and future
    top_axis.plot([axis_low, axis_high], [axis_low, axis_high],color="white", linewidth=1, linestyle="--")

    bottom_axis.axhline(0, color="white", linewidth=1, linestyle="--")

    slope, intercept = fits[scale]
    line_x = np.array([axis_low, axis_high])
    bottom_axis.plot(line_x, slope * line_x + intercept,color="red", linewidth=1.2)

    top_axis.set_title("Scale = " + str(scale))
    bottom_axis.set_xlabel("Novelty")

axes[0, 0].set_ylabel("Transience")
axes[1, 0].set_ylabel("Resonance")

fig.colorbar(mesh, ax=axes, label="counts", shrink=0.8)

fig.savefig(figure_png, dpi=300, bbox_inches="tight")

print("saved:", figure_png)