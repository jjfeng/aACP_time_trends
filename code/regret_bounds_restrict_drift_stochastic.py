import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def get_regret_bounds(max_loss, alpha, m, T, delta, drift, lambdas):
    # Using our regret bounds
    denom = -(
            (np.exp(-lambdas * (delta + drift)/max_loss) - 1)/((delta +
                drift)/max_loss) * (1 - alpha)
            + (np.exp(-lambdas) - 1) * alpha)
    raw_bound = -np.log(np.exp(-lambdas * delta * T)/max_loss + (m - 1) * np.exp(-lambdas
        * (delta + drift)/max_loss * T)) + np.log(m)
    bounds = raw_bound/denom / T
    return bounds * max_loss

def main(args=sys.argv[1:]):
    max_loss = 1
    n = 1000
    m = 10
    T = 50
    tau = np.sqrt(np.log(m))
    sigma_max = 0.5
    lambdas = np.exp(np.arange(-6, 2, 0.05))
    deltas = np.arange(0.03, min(max_loss, 0.3), 0.03)
    alphas = np.arange(0.5, 1, 0.1)
    index = 130
    alpha = 0.05
    for delta in deltas:
        drift = delta/2
        baseline_weight = 1/m
        bounds = get_regret_bounds(max_loss, alpha, m, T, delta, drift, lambdas)
        best_bound = np.min(bounds)
        best_idx = np.argmin(bounds)
        best_lambda = lambdas[best_idx]

        closest_idx = np.argmin(np.abs(bounds - 0.3))
        print("%.3f" % delta, "%.3f" % (delta + drift), "%.3f" % best_bound)
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

if __name__ == "__main__":
    main(sys.argv[1:])
