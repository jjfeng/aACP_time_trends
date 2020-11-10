from typing import List, Tuple
import logging
from itertools import product
from scipy import special
import scipy.stats
import numpy as np

from policy import Policy
from common import score_mixture_model


class ValidationPolicy(Policy):
    """
    Fixed Share
    First expert is human
    enforce monotonicity in the expert seq
    """

    def __init__(
        self,
        num_experts: int,
        etas: np.ndarray,
        human_max_loss: float,
        ci_alpha: float = 0.1,
        const_baseline_weight: float = 1e-10,
        num_back_batches: int = 3,
        ni_margin: float = 0,
    ):
        self.human_max_loss = human_max_loss
        self.ni_margin = ni_margin

        self.etas = etas[:-2]
        self.alpha = etas[-2]
        self.baseline_alpha = etas[-1]
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with only human weight
        self.baseline_weights = np.ones(1) * (self.baseline_alpha > 0)
        self.baseline_optim_weights = np.ones(1) * (self.baseline_alpha > 0)
        self.const_baseline_weight = np.ones(1) * const_baseline_weight
        self.const_baseline_optim_weight = np.ones(1) * const_baseline_weight

        self.ci_alpha = ci_alpha
        self.batch_sizes = np.array([])
        # print("self.const_baseline_weight", self.const_baseline_weight)

        self.loss_histories = np.zeros((num_experts, 0))
        self.var_loss_histories = np.zeros((num_experts, 0))

        self.num_back_batches = num_back_batches

        self.weight_history = []

    def __str__(self):
        return "ValidationPolicy"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        if self.curr_num_experts == 1:
            self.weights = np.ones(1) * (self.alpha > 0)
            self.optim_weights = np.ones(1) * (self.alpha > 0)
        else:
            self.baseline_weights = np.concatenate([self.baseline_weights, [0]])
            self.weights = np.concatenate([self.weights, [0]])
            self.baseline_optim_weights = np.concatenate(
                [self.baseline_optim_weights, [0]]
            )
            self.optim_weights = np.concatenate([self.optim_weights, [0]])

    def update_weights(
        self, time_t, criterion, batch_preds: np.ndarray, targets: np.ndarray,
        new_model_losses: np.ndarray
    ):
        if batch_preds is None:
            return

        indiv_robot_loss_t = np.array(
            [criterion(batch_preds[i, :], targets) for i in range(batch_preds.shape[0])]
        )
        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        var_model_losses_t = np.var(indiv_robot_loss_t, axis=1)
        if self.curr_num_experts > model_losses_t.size:
            model_losses_t = np.concatenate([model_losses_t, [0]])
            var_model_losses_t = np.concatenate([var_model_losses_t, [0]])
        new_losses = np.concatenate(
            [model_losses_t, [0] * (self.num_experts - model_losses_t.size),]
        ).reshape((-1, 1))
        var_new_losses = np.concatenate(
            [var_model_losses_t, [0] * (self.num_experts - model_losses_t.size),]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        self.var_loss_histories = np.concatenate(
            [self.var_loss_histories, var_new_losses], axis=1
        )
        assert self.loss_histories.shape == self.var_loss_histories.shape
        self.batch_sizes = np.concatenate(
            [self.batch_sizes, [indiv_robot_loss_t.shape[1]]]
        )
        v_weights = self.weights * np.exp(-self.etas[0] * model_losses_t)
        # print(self.weights, model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(
            -self.etas[0] * self.human_max_loss * np.ones(self.baseline_weights.size)
        )

        # heavily weight the new models when transitioning away
        # matrix11: model to model
        transition_matrix11 = np.eye(self.weights.size) * (1 - self.alpha)
        for i in range(self.weights.size - 1):
            transition_matrix11[i, i] = 1 - self.alpha - self.baseline_alpha
            transition_matrix11[i, i + 1 :] = (self.alpha) / (self.weights.size - i - 1)
            # transition_matrix11[i, :] = (self.alpha) / (self.weights.size - 1)
            # transition_matrix11[i, i] = 1 - self.alpha - self.baseline_alpha
        # matrix12: model to baseline
        transition_matrix12 = (
            np.eye(self.weights.size, self.baseline_weights.size, k=1)
            * self.baseline_alpha
        )
        # matrix21: baseline to model
        transition_matrix21 = np.zeros((self.baseline_weights.size, self.weights.size))
        transition_matrix21[0, -1] = self.alpha
        for i in range(1, self.weights.size - 1):
            transition_matrix21[i, i + 1 :] = (self.alpha) / (self.weights.size - i - 1)
            # transition_matrix21[i,-3:] = (self.alpha)/(3)
            # transition_matrix21[i,-1] = self.alpha
        # matrix22: baseline to baseline
        transition_matrix22 = np.eye(self.baseline_weights.size) * (1 - self.alpha)
        transition_matrix = np.block(
            [
                [transition_matrix11, transition_matrix12],
                [transition_matrix21, transition_matrix22],
            ]
        )
        # print(transition_matrix)
        transition_matrix[:, -1] = 1 - transition_matrix[:, :-1].sum(axis=1)

        # print("TRAN", transition_matrix)
        assert not np.any(np.isnan(transition_matrix))
        assert np.all(np.isclose(1, np.sum(transition_matrix, axis=1)))

        combo_vector = np.concatenate([v_weights, v_baseline_weights])
        new_combo_weights = np.matmul(
            combo_vector.reshape((1, -1)), transition_matrix
        ).flatten()
        self.weights = new_combo_weights[: self.weights.size]
        self.baseline_weights = np.maximum(new_combo_weights[self.weights.size :], 0)
        self.const_baseline_weight *= np.exp(-self.etas[0] * self.human_max_loss)

        # Adding normalization to prevent numerical underflow
        normalize_weights = special.softmax(
            np.concatenate(
                [
                    np.log(self.weights),
                    np.log(self.baseline_weights),
                    np.log(self.const_baseline_weight),
                ]
            )
        )
        self.weights = normalize_weights[: self.weights.size]
        self.baseline_weights = normalize_weights[
            self.weights.size : self.weights.size + self.baseline_weights.size
        ]
        self.const_baseline_weight = normalize_weights[-1:]

        self.weights = np.maximum(self.weights, 0)
        self.baseline_weights = np.maximum(self.baseline_weights, 0)
        self.const_baseline_weight = np.maximum(self.const_baseline_weight, 0)

        # Create predictions for future performance
        num_batches_list = (
            np.concatenate(
                [
                    np.ones(time_t - self.num_back_batches) * self.num_back_batches,
                    np.arange(self.num_back_batches, 0, step=-1),
                    [1],
                ]
            )
            if (time_t > self.num_back_batches)
            else np.concatenate([np.arange(time_t, 0, step=-1), [1]])
        ).astype(int)
        # estimate expected risk across mixture
        mean_loss = (
            np.sum(self.loss_histories[: time_t + 1, -self.num_back_batches :], axis=1)
            / num_batches_list
        )
        # get standard error of our estimate for expected risk across mixture
        var_loss_histories = self.var_loss_histories[
            : time_t + 1, -self.num_back_batches :
        ]
        raw_mean_vars = var_loss_histories / self.batch_sizes[-self.num_back_batches :]
        var_list = np.sum(raw_mean_vars *
                np.power(1/num_batches_list.reshape((-1,1)), 2),
                axis=1)

        # Note the bonferonni correction
        pred_t_factor = scipy.stats.norm.ppf(1 - self.ci_alpha / mean_loss.size)
        inflation = pred_t_factor * np.sqrt(var_list)
        # Predictions using the mean
        predictions = mean_loss + inflation
        predictions[-1] = np.mean(new_model_losses) + pred_t_factor * np.sqrt(np.var(new_model_losses)/new_model_losses.size)

        all_optim_weights = special.softmax(
            np.concatenate(
                [
                    np.log(self.weights) - self.etas[1] * predictions,
                    np.log(self.baseline_weights) - self.etas[1] * self.human_max_loss,
                    np.log(self.const_baseline_weight)
                    - self.etas[1] * self.human_max_loss,
                ]
            )
        )
        self.optim_weights = all_optim_weights[: self.optim_weights.size]
        self.baseline_optim_weights = all_optim_weights[
            self.optim_weights.size : self.optim_weights.size
            + self.baseline_weights.size
        ]
        self.const_baseline_optim_weight = all_optim_weights[-1:]

        # Impose constraint
        self.optim_weights *= predictions <= (
            self.human_max_loss + self.ni_margin
        )

    def get_predict_weights(self, time_t: int):
        denom = (
            np.sum(self.optim_weights)
            + np.sum(self.baseline_optim_weights)
            + self.const_baseline_optim_weight
        )
        robot_optim_weights = self.optim_weights / denom
        baseline_weight = 1 - np.sum(robot_optim_weights)
        self.weight_history.append([robot_optim_weights, baseline_weight])
        return robot_optim_weights, baseline_weight


class MetaExpWeightingList(Policy):
    """
    Meta exponential weighting list
    """

    def _get_policy_etas(self, etas):
        return np.array(etas)

    def __init__(
        self,
        eta: float,
        eta_list,
        meta_weights: np.ndarray,
        num_experts: int,
        human_max_loss: float,
        ci_alpha: float = 0.05,
        num_back_batches: int = 3,
        ni_margin: float = 0,
    ):
        self.eta = eta
        self.eta_list = eta_list
        self.eta_list_size = len(eta_list)

        self.eta_indexes = np.arange(len(eta_list))
        self.policy_dict = {}
        for idx, etas in enumerate(eta_list):
            self.policy_dict[etas] = ValidationPolicy(
                num_experts,
                np.array(etas),
                human_max_loss,
                ci_alpha=ci_alpha,
                num_back_batches=num_back_batches,
                ni_margin=ni_margin,
            )

        self.loss_ts = np.zeros(len(eta_list))
        self.human_max_loss = human_max_loss
        self.num_experts = num_experts

        self.meta_weights = meta_weights

    def __str__(self):
        return "Learning-to-Approve-%d" % len(self.eta_list)

    def add_expert(self, time_t):
        for k, policy in self.policy_dict.items():
            policy.add_expert(time_t)

    def _get_policy_prev_loss(
        self,
        time_t: int,
        criterion,
        batch_preds: np.ndarray,
        target: np.ndarray,
        policy: Policy,
    ):
        assert time_t == (len(policy.weight_history) - 1)

        robot_weights, human_weight = policy.weight_history[time_t]
        assert np.isclose(robot_weights.sum() + human_weight, 1)
        policy_loss = score_mixture_model(
            human_weight,
            robot_weights,
            criterion,
            batch_preds,
            target,
            self.human_max_loss,
        )
        return policy_loss

    def update_weights(
        self, time_t, criterion, batch_preds: np.ndarray, target: np.ndarray,
        new_model_losses: np.ndarray
    ):
        if batch_preds is not None:
            # Update the meta policy weights first
            loss_t = np.zeros(self.eta_list_size)
            for idx, etas in enumerate(self.eta_list):
                loss_t[idx] = self._get_policy_prev_loss(
                    time_t - 1, criterion, batch_preds, target, self.policy_dict[etas]
                )
                #print("policy loss", etas, loss_t[idx], self.loss_ts[idx] + loss_t[idx])
            self.loss_ts += loss_t
            self.meta_weights = self.meta_weights * np.exp(-self.eta * loss_t)

        # Let each policy update their own weights
        for policy in self.policy_dict.values():
            policy.update_weights(time_t, criterion, batch_preds, target,
                    new_model_losses)

    def get_predict_weights(self, time_t):
        denom = np.sum(self.meta_weights)
        policy_weights = self.meta_weights / denom

        robot_weights = 0
        human_weight = 0
        biggest_weight = 0
        biggest_eta = None
        for idx, etas in enumerate(self.eta_list):
            policy = self.policy_dict[etas]
            policy_eta = self._get_policy_etas(etas)
            policy_weight = policy_weights[idx]
            policy_robot_weights, policy_human_weight = policy.get_predict_weights(
                time_t
            )
            robot_weights += policy_robot_weights * policy_weight
            human_weight += policy_human_weight * policy_weight
            if biggest_weight < policy_weights[idx]:
                biggest_weight = policy_weights[idx]
                biggest_eta = policy_eta
            logging.info(
                "policy %s policy weight %.4f human weight %.4f top model %d top model weight %.4f avg model %.2f",
                policy_eta,
                policy_weight,
                policy_human_weight,
                np.argmax(policy_robot_weights),
                np.max(policy_robot_weights),
                np.sum(policy_robot_weights * np.arange(policy_robot_weights.size)),
            )
        logging.info("ETAS %s %.4f", biggest_eta, biggest_weight)
        print(
            "time",
            time_t,
            "max weigth robot",
            np.argmax(robot_weights),
            robot_weights.max(),
            "human",
            human_weight,
        )
        return robot_weights, human_weight
