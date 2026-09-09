import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

scales = [1, 60, 250, 1000, 7500] #must match NTR

grid = 150
clip = 0.1 #trimmed from each tail
colour_max = None #as Barron
#per_row = 3 #scales per block
positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)] #block, col

source = r"G:\My Drive\Birkbeck\Project\Hansard"
ntr_csv = os.path.join(source, "hansard_ntr_by_scale.csv")
figure_png = os.path.join(source, "density_TvN_RvN.png")
ntr = pd.read_csv(ntr_csv)
print("loaded:", ntr.shape)

for scale in scales:
    if "novelty_" + str(scale) not in ntr.columns:
        raise SystemExit("no columns for scale " + str(scale) + " - rerun script 9") #check

#=========one panel pair per scale==========================
fig, axes = plt.subplots(4, 3, figsize=(12, 16), constrained_layout=True)

for index in range(len(scales)):
    scale = scales[index]
    block, column = positions[index]

    frame = ntr[["novelty_" + str(scale), "transience_" + str(scale), "resonance_" + str(scale)]].dropna()

    novelty = frame["novelty_" + str(scale)].to_numpy()
    transience = frame["transience_" + str(scale)].to_numpy()
    resonance = frame["resonance_" + str(scale)].to_numpy()

    axis_low = min(np.percentile(novelty, clip), np.percentile(transience, clip)) #same limit
    axis_high = max(np.percentile(novelty, 100 - clip), np.percentile(transience, 100 - clip))
    resonance_bound = np.percentile(np.abs(resonance), 100 - clip) #centred on zero

    top, x_edges, y_edges_top = np.histogram2d(novelty, transience, bins=grid, range=[[axis_low, axis_high], [axis_low, axis_high]])
    bottom, x_edges, y_edges_bottom = np.histogram2d(novelty, resonance, bins=grid, range=[[axis_low, axis_high], [-resonance_bound, resonance_bound]])
    slope, intercept = np.polyfit(novelty, resonance, 1)

    if colour_max:
        panel_max = colour_max
    else:
        panel_max = max(top.max(), bottom.max())

    colour_scale = LogNorm(vmin=1, vmax=panel_max)

    top_axis = axes[block * 2, column]
    bottom_axis = axes[block * 2 + 1, column]

    mesh = top_axis.pcolormesh(x_edges, y_edges_top, np.ma.masked_where(top == 0, top).T, norm=colour_scale, cmap="viridis")
    bottom_axis.pcolormesh(x_edges, y_edges_bottom, np.ma.masked_where(bottom == 0, bottom).T, norm=colour_scale, cmap="viridis")

    top_axis.plot([axis_low, axis_high], [axis_low, axis_high], color="white", linewidth=1, linestyle="--")
    bottom_axis.axhline(0, color="white", linewidth=1, linestyle="--")
    line_x = np.array([axis_low, axis_high])
    bottom_axis.plot(line_x, slope * line_x + intercept, color="red", linewidth=1.2)

    top_axis.set_title("Scale = " + str(scale) + "\nslope " + str(round(slope, 3)))
    bottom_axis.set_xlabel("Novelty")

    print("scale:", str(scale).ljust(6), "points: ", len(novelty),"Slope: ", round(slope, 5),"Busiest cell: ", int(panel_max))

axes[0, 0].set_ylabel("Transience")
axes[1, 0].set_ylabel("Resonance")
axes[2, 0].set_ylabel("Transience")
axes[3, 0].set_ylabel("Resonance")

axes[2, 2].axis("off")
axes[3, 2].axis("off")

colour_bar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.6)
colour_bar.set_ticks([])
colour_bar.set_label("Speech count, low to high")

fig.savefig(figure_png, dpi=300)
plt.close(fig)

print("saved:", figure_png)