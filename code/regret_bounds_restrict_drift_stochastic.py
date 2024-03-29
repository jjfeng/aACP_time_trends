import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

from regret_bounds_small_loss import get_small_loss_bounds

def _get_bernstein_factor(max_val, lambdas):
    assert np.all(max_val <= 1)
    return (1 - np.exp(-max_val * lambdas)) / max_val


def get_regret_bounds(
    meta_weights, T, delta, drift, ni_margin, lambdas, n, alpha, sigma_max=0.5
):
    assert np.isclose(np.sum(meta_weights), 1)
    lambdas = lambdas.reshape((1, -1))
    eps = np.exp(np.arange(-3, 0, 0.02))
    _alpha3 = T * np.exp(-2 * np.power(eps, 2) * np.power(sigma_max, 2) * n)
    _alpha3 = (T - 1) * np.exp(-2 * np.power(eps, 2) * np.power(sigma_max, 2) * n) + np.exp(-2 * np.power(eps, 2) * np.power(sigma_max, 2) * n/2)

    eps_mask = ((delta + ni_margin + drift + eps * sigma_max) < 1) & (
        _alpha3 + alpha < 1
    )
    eps = eps[eps_mask].reshape((-1, 1))

    factor1 = _get_bernstein_factor(delta + ni_margin + drift, lambdas)
    factor2 = _get_bernstein_factor(
        delta + ni_margin + drift + eps * sigma_max, lambdas
    )
    factor3 = _get_bernstein_factor(1, lambdas)
    alpha2 = alpha
    alpha3 = T * np.exp(-2 * np.power(eps, 2) * np.power(sigma_max, 2) * n)
    alpha3 = (T - 1) * np.exp(-2 * np.power(eps, 2) * np.power(sigma_max, 2) * n) + np.exp(-2 * np.power(eps, 2) * np.power(sigma_max, 2) * n/2)
    #print(alpha3_o, alpha3)
    alpha1 = 1 - alpha3 - alpha2
    factor = 1 / (factor1 * alpha1 + factor2 * alpha2 + factor3 * alpha3)

    baseline_weight = meta_weights[0]
    raw_bound = -np.log(
        baseline_weight * np.exp(-lambdas * delta * T)
        + (1 - baseline_weight) * np.exp(-lambdas * T)
    )
    error = np.power(lambdas, 2) / 8 / n * T

    bounds = (raw_bound + error) * factor / T
    best_bound = np.min(bounds)
    best_idx = np.where(bounds == np.min(bounds))
    best_eps_idx = best_idx[0][0]
    best_lambda_idx = best_idx[1][0]
    # print("best eps", eps.flatten()[best_eps_idx], "vs", 1/np.sqrt(n))
    bounds = np.min(bounds, axis=0)
    return bounds

def get_restrict_bounds(lambdas, deltas, m):
    NI_FACTOR = 0.1
    n = 10000
    T = 50
    alpha = 0.1
    delta_bounds = {}
    for delta in deltas:
        print("======================")
        print("%.3f" % delta)
        drift = delta * 2
        meta_weights = np.ones(m) / m
        bounds = get_regret_bounds(
            meta_weights,
            T,
            delta,
            drift,
            ni_margin=delta * NI_FACTOR,
            lambdas=lambdas,
            n=n,
            alpha=alpha,
        )
        best_bound = np.min(bounds)
        best_idx = np.argmin(bounds)
        best_lambda = lambdas[best_idx]
        print(delta, best_bound, best_lambda)
        print(delta, delta * 2, lambdas[np.max(np.where(bounds < delta * 2)[0])])
        delta_bounds[delta] = bounds
    return delta_bounds

def main(args=sys.argv[1:]):
    max_loss = 1
    m = 10
    lambdas = np.exp(np.arange(-6, 2, 0.05))
    deltas = np.arange(0.03, min(max_loss, 0.2), 0.03)

    delta_bounds_restrict = get_restrict_bounds(lambdas, deltas, m)
    delta_bounds_small = get_small_loss_bounds(lambdas, deltas, m, max_loss)

    sns.set_context("paper", font_scale=1.5)

    colors = []
    for delta in deltas:
        bounds = delta_bounds_restrict[delta]
        p = plt.plot(lambdas, bounds, label=r"$\delta$=%.2f" % delta)
        colors.append(p[0].get_color())
    for delta, color in zip(deltas, colors):
        bounds = delta_bounds_small[delta]
        plt.plot(lambdas, bounds, '--', color=color)


    # Add X and y Label#
    plt.ylim(0, min(1, max_loss * 2))
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Average risk bound")

    # Add a Legend
    plt.legend(loc="upper left")

    plt.savefig("_output/avg_loss_bounds_drift_restrict_stoch_%d.png" % m)
    print("_output/avg_loss_bounds_drift_restrict_stoch_%d.png" % m)


if __name__ == "__main__":
    main(sys.argv[1:])
