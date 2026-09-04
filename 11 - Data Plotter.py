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

