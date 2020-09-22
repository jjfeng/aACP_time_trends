import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

batch_size = 100
max_loss = 0.5
m = 10
T = 50
lambdas = np.exp(np.arange(-6, 2, 0.05))
deltas = np.arange(0.03, 0.3, 0.03)
num_wrongs = np.arange(0.1, 3, 0.2)
# velocity of drift
v = 0.05
delta = deltas[2]
for num_wrong in num_wrongs:
    t_alpha = norm.ppf(1 - num_wrong / T)
    se_est = t_alpha * np.sqrt(delta * (1 - delta) / batch_size)
    constraint = delta + se_est + v
    actually_worst_case = (constraint * (T - num_wrong) + num_wrong) / T
    bounds = (
        constraint
        / (1 - np.exp(-constraint * lambdas))
        * (
            lambdas * delta * T
            + np.log(m)
            - np.log(
                1
                + (m - 1)
                * np.exp(
                    -lambdas
                    * ((T - num_wrong) * constraint - delta * T + max_loss * num_wrong)
                )
            )
            + (np.exp(-max_loss * lambdas) - 1) / max_loss * num_wrong
            - (np.exp(-lambdas * constraint) - 1) / constraint * num_wrong
        )
    )
    best_bound = np.min(bounds) / T
    best_lambda = lambdas[np.argmin(bounds)]
    # Create the plot
    plt.plot(
        lambdas,
        bounds / T,
        label="d=%.2f, bound=%.2f, lam=%.5f, nwrong=%.1f"
        % (delta, best_bound, best_lambda, num_wrong),
    )
    print("num wrong", num_wrong)
    print(
        "d=%.2f, bound=%.5f, lam=%.5f, worst=%.2f, c=%.2f"
        % (delta, best_bound, best_lambda, actually_worst_case, constraint)
    )

# Add X and y Label#
# plt.ylim(0, Y_MAX)
plt.xlabel("lambda")
plt.ylabel("average loss bound")

# Add a Legend
plt.legend()

plt.savefig("_output/avg_loss_bounds.png")
