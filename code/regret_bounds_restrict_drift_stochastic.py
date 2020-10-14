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
inflation = tau * sigma_max / np.sqrt(n)
print("inflation", inflation)
lambdas = np.exp(np.arange(-6, 2, 0.05))
deltas = np.arange(0.03, min(max_loss, 0.3), 0.03)
alphas = np.arange(0.5, 1, 0.1)
index = 130
alpha = 0.05
for delta in deltas:
    drift = delta
    denom = -(np.exp(-lambdas * (delta + drift)) - 1)/(delta + drift) * (1 - alpha) + (np.exp(-lambdas) - 1) * alpha
    print(denom[0])
    raw_bound = lambdas * delta * T + np.log(m)
    bounds = raw_bound/denom / T
    best_bound = np.min(bounds)
    best_idx = np.argmin(bounds)
    best_lambda = lambdas[best_idx]
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

plt.savefig("_output/avg_loss_bounds_drift_restrict_stoch_%d.png" % m)
print("_output/avg_loss_bounds_drift_restrict_stoch_%d.png" % m)
