import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

drift = 0.1

max_loss = 1
n = 100
m = 10
baseline_weight = 1/m
T = 50
tau = np.sqrt(np.log(T))
sigma_max = 0.5
inflation = tau * sigma_max/np.sqrt(n)
print("inflation", inflation)
lambdas = np.exp(np.arange(-6, 2, 0.05))
deltas = np.arange(0.03, min(max_loss, 0.3), 0.03)
for delta in deltas:
    c = min(delta + drift + 2 * inflation, 1)
    bounds = (
        c/ (1 - np.exp(-lambdas * c))
        * (-np.log(np.exp(-lambdas * delta * T) + (m - 1) * np.exp(-lambdas * c * T)) + np.log(m) + lambdas * inflation * T)
    )/T
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

plt.savefig("_output/avg_loss_bounds_drift_restrict_stoch.png")
print("_output/avg_loss_bounds_drift_restrict_stoch.png")
