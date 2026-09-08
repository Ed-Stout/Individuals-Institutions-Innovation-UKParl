import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

scales = [1, 60, 250, 1000, 7500] #must match the NTR script

grid = 150
clip = 0.1 #percent trimmed from each tail
colour_max = None #None = each panel scaled to its own busiest cell, as Barron. Set a number to share.

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
figure_png = os.path.join(source, "density_TvN_RvN.png")

ntr = pd.read_csv(ntr_csv)
print("loaded:", ntr.shape)

#catches scripts 9 and 11 drifting apart if scales change in one and not the other
for scale in scales:
    if "novelty_" + str(scale) not in ntr.columns:
        raise SystemExit("no columns for scale " + str(scale) + " - rerun script 9")

#=========one panel pair per scale==========================
fig, axes = plt.subplots(2, len(scales), figsize=(4 * len(scales), 8), constrained_layout=True) #constrained layout to 

for column in range(len(scales)):
    scale = scales[column]

    frame = ntr[["novelty_" + str(scale), "transience_" + str(scale), "resonance_" + str(scale)]].dropna()

    novelty = frame["novelty_" + str(scale)].to_numpy()
    transience = frame["transience_" + str(scale)].to_numpy()
    resonance = frame["resonance_" + str(scale)].to_numpy()

    #trimmed rather than min/max so a few extreme speeches don't stretch the axis
    axis_low = min(np.percentile(novelty, clip), np.percentile(transience, clip)) #novelty and transience share limits
    axis_high = max(np.percentile(novelty, 100 - clip), np.percentile(transience, 100 - clip))
    resonance_bound = np.percentile(np.abs(resonance), 100 - clip) #centred on zero

    top, x_edges, y_edges_top = np.histogram2d(novelty, transience, bins=grid, range=[[axis_low, axis_high], [axis_low, axis_high]])

    bottom, x_edges, y_edges_bottom = np.histogram2d(novelty, resonance, bins=grid, range=[[axis_low, axis_high], [-resonance_bound, resonance_bound]])

    #raw OLS slope, NOT gamma - script 10 fits gamma on z-scores
    slope, intercept = np.polyfit(novelty, resonance, 1)

    if colour_max:
        panel_max = colour_max
    else:
        panel_max = max(top.max(), bottom.max())

    colour_scale = LogNorm(vmin=1, vmax=panel_max)

    top_axis = axes[0, column]
    bottom_axis = axes[1, column]

    #empty cells masked so they render as background, not as the lowest colour
    #.T because histogram2d returns x down the rows, pcolormesh wants x across
    mesh = top_axis.pcolormesh(x_edges, y_edges_top, np.ma.masked_where(top == 0, top).T, norm=colour_scale, cmap="viridis")

    bottom_axis.pcolormesh(x_edges, y_edges_bottom, np.ma.masked_where(bottom == 0, bottom).T, norm=colour_scale, cmap="viridis")

    #identity line - speeches on it are equally surprising to past and future
    top_axis.plot([axis_low, axis_high], [axis_low, axis_high], color="white", linewidth=1, linestyle="--")

    bottom_axis.axhline(0, color="white", linewidth=1, linestyle="--")

    line_x = np.array([axis_low, axis_high])
    bottom_axis.plot(line_x, slope * line_x + intercept, color="red", linewidth=1.2)

    top_axis.set_title("Scale = " + str(scale) + "\nslope " + str(round(slope, 3)))
    bottom_axis.set_xlabel("Novelty")

    #only the last colorbar is labelled - colour is normalised within each panel, not across them
    if column == len(scales) - 1:
        fig.colorbar(mesh, ax=[top_axis, bottom_axis], label="counts (per-panel scale)", shrink=0.6)
    else:
        fig.colorbar(mesh, ax=[top_axis, bottom_axis], shrink=0.6)

    print("scale", str(scale).ljust(6), "points", len(novelty),
          "| raw slope", round(slope, 5),
          "| busiest cell", int(panel_max))

axes[0, 0].set_ylabel("Transience")
axes[1, 0].set_ylabel("Resonance")

fig.savefig(figure_png, dpi=300)
plt.close(fig)

print("saved:", figure_png)