import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

drift = 0.1

max_loss = 1
baseline_weight = 0.25
T = 50
lambdas = np.exp(np.arange(-6, np.log(0.25 / drift), 0.05))
deltas = np.arange(0.03, min(max_loss, 0.3), 0.03)
for delta in deltas:
    bounds = (
        delta
        + (2 * lambdas * T * (drift ** 2) + np.log(1 / baseline_weight) / lambdas) / T
    )
    best_bound = np.min(bounds)
    best_lambda = lambdas[np.argmin(bounds)]
    # Create the plot
    plt.plot(
        lambdas,
        bounds,
        label="d=%.2f, bound=%.3f, lam=%.3f" % (delta, best_bound, best_lambda),
    )

# Add X and y Label#
plt.ylim(0, min(1, max_loss * 2))
plt.xlabel("lambda")
plt.ylabel("average loss bound")

# Add a Legend
plt.legend()

plt.savefig("_output/avg_loss_bounds_optim.png")
print("_output/avg_loss_bounds_optim.png")
