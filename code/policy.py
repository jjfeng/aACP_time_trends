import logging
from typing import List, Dict, Tuple

import scipy.stats
import numpy as np

from proposer import PredsTarget


class Policy:
    def __init__(self, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def get_predict_weights(self, time_t: int):
        raise NotImplementedError()

    def update_weights(
        self,
        time_t,
        criterion,
        batch_model_preds_targets: PredsTarget = None,
        holdout_preds_target: PredsTarget = None,
    ):
        return

    def predict_next_losses(self, time_t: int):
        return np.zeros(self.curr_num_experts)

    @property
    def is_oracle(self):
        return False


class BaselinePolicy(Policy):
    def __str__(self):
        return "Baseline"

    def get_predict_weights(self, time_t: int):
        return np.zeros(self.curr_num_experts), 1


class FixedPolicy(Policy):
    def __str__(self):
        return "Fixed"

    def get_predict_weights(self, time_t: int):
        a = np.zeros(self.curr_num_experts)
        a[0] = 1
        return a, 0


class BlindApproval(Policy):
    def __str__(self):
        return "Blind"

    def __init__(self, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def get_predict_weights(self, time_t: int):
        a = np.zeros(self.curr_num_experts)
        # if a.size > 2:
        #    a[-2] = 1
        # else:
        a[-1] = 1
        print(time_t, "chosen robot", np.where(a))
        return a, 0


class OracleApproval(Policy):
    def __str__(self):
        return "Oracle"

    def __init__(self, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def get_predict_weights(self, time_t: int):
        a = np.zeros(self.curr_num_experts)
        return a, 0

    @property
    def is_oracle(self):
        return True


class TTestApproval(Policy):
    def __str__(self):
        return "T-Test"

    def __init__(
        self,
        num_experts: int,
        human_max_loss: float,
        ci_alpha: float = 0.025,
        ni_margin: float = 0,
    ):
        self.human_max_loss = human_max_loss
        self.ni_margin = ni_margin
        self.curr_num_experts = 0
        self.num_experts = num_experts
        self.loss_histories = [[] for i in range(self.num_experts)]
        self.curr_approved_idx = 0
        assert ci_alpha < 0.5
        self.factor = scipy.stats.norm().ppf(1 - ci_alpha)
        print(self.factor)
        assert self.factor > 0
        self.robot_weights = np.array([1])
        self.human_weight = 0

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def _get_upper_ci_risk(self, new_model_loss):
        """
        @return upper ci of the model's risk given an array of losses
        """
        upper_ci_risk = np.mean(new_model_loss) + self.factor * np.sqrt(
            np.var(new_model_loss) / new_model_loss.size
        )
        return upper_ci_risk

    def _get_upper_ci_diff(self, new_model_loss, baseline_model_loss):
        loss_improvement = new_model_loss - baseline_model_loss
        mean_improve = np.mean(loss_improvement)
        # upper ci of difference
        diff_ci_bound = mean_improve + self.factor * np.sqrt(
            np.var(loss_improvement) / new_model_loss.size
        )
        return diff_ci_bound

    def _can_approve(self, upper_ci_risk, diff_ci_bound):
        return (upper_ci_risk <= (self.human_max_loss + self.ni_margin)) and (diff_ci_bound <= 0)

    def update_weights(
        self,
        time_t,
        criterion,
        batch_model_preds_target: PredsTarget = None,
        holdout_pred_target: PredsTarget = None,
    ):
        if batch_model_preds_target is None:
            return

        indiv_robot_loss_t = np.array(
            [
                criterion(batch_model_preds_target.preds[i, :],
                    batch_model_preds_target.target)
                for i in range(batch_model_preds_target.preds.shape[0])
            ]
        )
        num_updates = min(self.curr_num_experts, indiv_robot_loss_t.shape[0])
        for i in range(num_updates):
            self.loss_histories[i].append(indiv_robot_loss_t[i, :])
        for i in range(num_updates, self.num_experts):
            self.loss_histories[i].append([])

        best_model_idx = self.curr_approved_idx
        # Check that the baseline model is not inferior to human
        best_upper_ci = self._get_upper_ci_risk(np.concatenate(self.loss_histories[self.curr_approved_idx][self.curr_approved_idx:]))
        best_can_approve = self._can_approve(best_upper_ci, 0)

        # Compare all other models to the baseline model
        best_diff_ci = 0
        for i in range(self.curr_approved_idx + 1, self.curr_num_experts - 1):
            new_model_loss = np.concatenate(self.loss_histories[i][i:])

            # upper ci compare to human + ni_margin
            upper_ci_risk = self._get_upper_ci_risk(new_model_loss)

            # upper ci compare to currently approved
            baseline_model_loss = np.concatenate(
                self.loss_histories[self.curr_approved_idx][i:]
            )
            diff_ci_bound = self._get_upper_ci_diff(new_model_loss, baseline_model_loss)
            if self._can_approve(upper_ci_risk, diff_ci_bound)and diff_ci_bound < best_diff_ci:
                best_model_idx = i
                best_diff_ci = diff_ci_bound
                best_can_approve = True

        # TTest the latest model also!
        new_model_loss = criterion(holdout_pred_target.preds[-1], holdout_pred_target.target)
        upper_ci_risk = self._get_upper_ci_risk(new_model_loss)
        baseline_model_loss = criterion(holdout_pred_target.preds[self.curr_approved_idx], holdout_pred_target.target)
        diff_ci_bound = self._get_upper_ci_diff(new_model_loss, baseline_model_loss)
        if self._can_approve(upper_ci_risk, diff_ci_bound) and (diff_ci_bound < best_diff_ci):
            print(diff_ci_bound, best_diff_ci)
            best_model_idx = self.curr_num_experts - 1
            best_can_approve = True
            print("USING THE LATEST MODEL", best_model_idx)

        if best_can_approve:
            self.curr_approved_idx = best_model_idx
            self.robot_weights = np.zeros(self.curr_num_experts)
            self.robot_weights[best_model_idx] = 1
            self.human_weight = 0
            logging.info("TTEST %d approved %d", time_t, self.curr_approved_idx)
        else:
            # worse than human
            self.robot_weights = np.zeros(self.curr_num_experts)
            self.human_weight = 1

    def get_predict_weights(self, time_t: int):
        return self.robot_weights, self.human_weight
