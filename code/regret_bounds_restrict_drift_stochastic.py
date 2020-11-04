import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def get_regret_bounds(meta_weights, T, delta, drift, lambdas, n=50, sigma_max=0.5):
    assert np.isclose(np.sum(meta_weights), 1)
    print("DRIFT", drift)
    baseline_weight = meta_weights[0]
    factor = (delta + drift)/(1 - np.exp(-lambdas * (delta + drift)))
    raw_bound = -np.log(
        baseline_weight * np.exp(-lambdas * delta * T) + (1 - baseline_weight) * np.exp(-lambdas * (delta + drift) * T)
    )
    standard_error = sigma_max / np.sqrt(n) / np.sqrt(T) * 1.9

    bounds = raw_bound * factor / T
    best_idx = np.argmin(bounds + standard_error)
    print("SE", standard_error, "bound", bounds[best_idx])
    return bounds + standard_error


def main(args=sys.argv[1:]):
    max_loss = 1
    n = 1000
    m = 10
    T = 50
    lambdas = np.exp(np.arange(-6, 2, 0.05))
    deltas = np.arange(0.03, min(max_loss, 0.2), 0.03)
    index = 130
    alpha = 0.05
    for delta in deltas:
        drift = delta * 2
        baseline_weight = 1 / m
        meta_weights = np.ones(m)/m
        bounds = get_regret_bounds(meta_weights, T, delta, drift, lambdas)
        best_bound = np.min(bounds)
        best_idx = np.argmin(bounds)
        best_lambda = lambdas[best_idx]

        print("%.3f" % delta)
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
