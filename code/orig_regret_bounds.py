import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

max_loss = 1
m = 6
baseline_weight = 0.5
T = 50
lambdas = np.exp(np.arange(-6, 2, 0.05))
deltas = np.arange(0.03, 0.3, 0.03)
for delta in deltas:
    bounds = max_loss/(1 - np.exp(-max_loss * lambdas)) * (
        lambdas * delta * T
        + np.log(1/baseline_weight)
        )
    best_bound = np.min(bounds)/T
    best_lambda = lambdas[np.argmin(bounds)]
    # Create the plot
    plt.plot(lambdas, bounds/T, label='d=%.2f, bound=%.2f, lam=%.3f' % (delta, best_bound, best_lambda))

# Add X and y Label#
plt.ylim(0,max_loss)
plt.xlabel('lambda')
plt.ylabel('average loss bound')

# Add a Legend
plt.legend()

plt.savefig("_output/avg_loss_bounds_orig.png")
