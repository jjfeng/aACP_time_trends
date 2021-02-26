import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def get_small_loss_bounds(lambdas, deltas, m, max_loss):
    baseline_weight = 1 / m
    T = 50
    delta_bounds = {}
    for delta in deltas:
        bounds = (
            max_loss
            / (1 - np.exp(-max_loss * lambdas))
            * (lambdas * delta * T + np.log(1 / baseline_weight))
        ) / T
        best_bound = np.min(bounds)
        best_lambda = lambdas[np.argmin(bounds)]
        delta_bounds[delta] = bounds

    return delta_bounds

def main(args=sys.argv[1:]):
    max_loss = 1
    m = 10
    lambdas = np.exp(np.arange(-6, 2, 0.05))
    deltas = np.arange(0.03, min(max_loss, 0.2), 0.03)
    delta_bounds = get_small_loss_bounds(lambdas, deltas, m, max_loss)
    for delta in deltas:
        bounds = delta_bounds[delta]
        # Create the plot
        plt.plot(
            lambdas, bounds, label="d=%.2f" % (delta),
        )

    # Add X and y Label#
    plt.ylim(0, min(1, max_loss * 2))
    plt.xlabel("Lambda")
    plt.ylabel("Average risk bound")

    # Add a Legend
    plt.legend()

    plt.savefig("_output/avg_loss_bounds_small.png")
    print("_output/avg_loss_bounds_small.png")


if __name__ == "__main__":
    main(sys.argv[1:])
