import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

drift = 0.1

max_loss = 1
m = 10
baseline_weight = 1 / m
T = 50
lambdas = np.exp(np.arange(-6, 2, 0.05))
index = 130
deltas = np.arange(0.03, min(max_loss, 0.3), 0.03)
for delta in deltas:
    bounds = (
        (delta + drift)
        / (1 - np.exp(-lambdas * (delta + drift)))
        * (
            lambdas * delta * T
            + np.log(1 / baseline_weight)
            - np.log(1 + (1 - baseline_weight) * np.exp(-lambdas * (delta + drift) * T))
        )
    ) / T
    best_bound = np.min(bounds)
    best_lambda = lambdas[np.argmin(bounds)]
    print(delta, lambdas[index], bounds[index])
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

plt.savefig("_output/avg_loss_bounds_drift_restrict.png")
print("_output/avg_loss_bounds_drift_restrict.png")
