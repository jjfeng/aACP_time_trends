import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def get_regret_bounds(meta_weights, T, delta, drift, lambdas, n=50, sigma_max=0.5):
    assert np.isclose(np.sum(meta_weights), 1)
    baseline_weight = meta_weights[0]
    factor = (delta + drift)/(1 - np.exp(-lambdas * (delta + drift)))
    raw_bound = -np.log(
        baseline_weight * np.exp(-lambdas * delta * T) + (1 - baseline_weight) * np.exp(-lambdas * (delta + drift) * T)
    )
    error = sigma_max**2 /2/n * np.power(lambdas, 2) * T

    bounds = (raw_bound + error) * factor / T
    best_idx = np.argmin(bounds)
    print("bound", bounds[best_idx])
    print("INFLA", bounds[best_idx]/delta)
    print("best lam", lambdas[best_idx])
    return bounds


def main(args=sys.argv[1:]):
    max_loss = 1
    n = 75
    m = 15
    T = 50
    lambdas = np.exp(np.arange(-6, 2, 0.05))
    deltas = np.arange(0.03, min(max_loss, 0.2), 0.03)
    index = 130
    alpha = 0.05
    for delta in deltas:
        print("======================")
        print("%.3f" % delta)
        drift = delta
        baseline_weight = 1 / m
        meta_weights = np.ones(m)/m
        bounds = get_regret_bounds(meta_weights, T, delta, drift, lambdas)
        best_bound = np.min(bounds)
        best_idx = np.argmin(bounds)
        best_lambda = lambdas[best_idx]

        # Create the plot
        plt.plot(lambdas, bounds, label="d=%.2f" % delta)

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
