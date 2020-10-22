import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

max_loss = 1
m = 10
baseline_weight = 1 / m
T = 50
lambdas = np.exp(np.arange(-6, 2, 0.05))
deltas = np.arange(0.03, min(max_loss, 0.2), 0.03)
for delta in deltas:
    bounds = (
        max_loss
        / (1 - np.exp(-max_loss * lambdas))
        * (lambdas * delta * T + np.log(1 / baseline_weight))
    ) / T
    best_bound = np.min(bounds)
    best_lambda = lambdas[np.argmin(bounds)]
    # Create the plot
    plt.plot(
        lambdas,
        bounds,
        label="d=%.2f" % (delta),
    )

# Add X and y Label#
plt.ylim(0, min(1, max_loss * 2))
plt.xlabel("lambda")
plt.ylabel("average loss bound")

# Add a Legend
plt.legend()

plt.savefig("_output/avg_loss_bounds_small.png")
print("_output/avg_loss_bounds_small.png")
