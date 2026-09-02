#Plot for LDA graph

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

save_path = Path(r"C:\Dissertation Project\LDA_output")

#=======parameters =========
topicnum = 100
refresh = 50

loglik = np.loadtxt(save_path / f'loglik_full_k{topicnum}.txt') #per line

iterations = []
for position in range(len(loglik)): #how far through iterations is it
    iterations.append(position * refresh)

plt.figure(figsize=(8, 5))
plt.plot(iterations, loglik, marker='o', markersize=3)
plt.xlabel("iterations")
plt.ylabel("Log likelihood") #how well fit is it, does it explain data
plt.title(f"Convergence, K={topicnum}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_path / f'convergence_k{topicnum}.png', dpi=200)
plt.show()

#=======gain per block, diminishing returns table=========
print("iteration, gain over previous block")
for position in range(1, len(loglik)):
    gain = loglik[position] - loglik[position - 1] #table showing how it improve
    print(iterations[position], round(float(gain)))