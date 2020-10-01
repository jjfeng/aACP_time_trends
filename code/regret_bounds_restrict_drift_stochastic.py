import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

max_loss = 1
n = 50
alpha = 0.05
m = 10
T = 50
tau = np.sqrt(np.log(m))
sigma_max = 0.5
inflation = tau * sigma_max/np.sqrt(n)
print("inflation", inflation)
lambdas = np.exp(np.arange(-6, 2, 0.05))
deltas = np.arange(0.03, min(max_loss, 0.3), 0.03)
index = 130
for delta in deltas:
    drift = delta
    c = min(delta + drift + 1 * inflation, 1)
    multiplier = 1/((1 - np.exp(-lambdas * c))/c * (1 - alpha) + (1 - np.exp(-lambdas)) * alpha)
    raw_bound = (-np.log(np.exp(-lambdas * delta * T) + (m - 1) * np.exp(-lambdas * c * T)) + np.log(m))
    bounds = (
        multiplier * raw_bound
    )/T
    best_bound = np.min(bounds)
    best_idx = np.argmin(bounds)
    best_lambda = lambdas[best_idx]
    #print(delta, lambdas[index], bounds[index])
    print(delta, "c", c, (1 - np.exp(-lambdas[best_idx] * c)))
    print(delta, multiplier[best_idx], raw_bound[best_idx]/T, 1/(1 - np.exp(-lambdas[best_idx])), c/(1 - np.exp(-lambdas[best_idx] * c)))
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
