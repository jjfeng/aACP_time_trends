import logging
from typing import List, Dict

import numpy as np


class Policy:
    def __init__(self, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def get_predict_weights(self, time_t: int):
        raise NotImplementedError()

    def update_weights(
        self, time_t, indiv_robot_loss_t=None, prev_weights=None, mixture_func=None
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

    def __init__(self, num_experts: int, human_max_loss: float, factor: float = 1.96):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0
        self.num_experts = num_experts
        self.loss_histories = [[] for i in range(self.num_experts)]
        self.curr_approved_idx = 0
        self.factor = factor

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def update_weights(
        self, time_t, indiv_robot_loss_t=None, prev_weights=None, mixture_func=None
    ):
        if indiv_robot_loss_t is None:
            return

        num_updates = min(self.curr_num_experts, indiv_robot_loss_t.shape[0])
        for i in range(num_updates):
            self.loss_histories[i].append(indiv_robot_loss_t[i, :])
        for i in range(num_updates, self.num_experts):
            self.loss_histories[i].append([])

    def get_predict_weights(self, time_t: int):
        best_model_idx = self.curr_approved_idx
        best_upper_ci = 0
        differences = []
        for i in range(self.curr_approved_idx + 1, self.curr_num_experts - 1):
            new_model_loss = np.concatenate(self.loss_histories[i][i:])
            baseline_model_loss = np.concatenate(
                self.loss_histories[self.curr_approved_idx][i:]
            )
            loss_improvement = new_model_loss - baseline_model_loss
            mean_improve = np.mean(loss_improvement)
            differences.append(mean_improve)
            upper_ci = mean_improve + self.factor * np.sqrt(
                np.var(loss_improvement) / new_model_loss.size
            )
            is_better = upper_ci < 0

            lower_ci_human_diff = np.mean(
                new_model_loss - self.human_max_loss
            ) - self.factor * np.sqrt(np.var(new_model_loss) / new_model_loss.size)
            is_worse_than_human = lower_ci_human_diff > 0

            if is_better and not is_worse_than_human and upper_ci < best_upper_ci:
                print("new model loss", np.mean(new_model_loss), self.human_max_loss)
                best_model_idx = i
                best_upper_ci = upper_ci

        a = np.zeros(self.curr_num_experts)
        if len(self.loss_histories[0]) == 0:
            return a, 1
        else:
            best_model_loss = np.concatenate(
                self.loss_histories[best_model_idx][best_model_idx:]
            )
            lower_ci_human_diff = np.mean(
                best_model_loss - self.human_max_loss
            ) - self.factor * np.sqrt(np.var(best_model_loss) / best_model_loss.size)
            print(
                "human diff lower ci",
                lower_ci_human_diff,
                np.mean(best_model_loss),
                self.human_max_loss,
                "se",
                np.sqrt(np.var(best_model_loss) / best_model_loss.size),
            )
            is_worse_than_human = lower_ci_human_diff > 0

            if is_worse_than_human:
                print("ttest human")
                return a, 1
            else:
                self.curr_approved_idx = best_model_idx
                a[best_model_idx] = 1
                logging.info("TTEST %d approved %d", time_t, self.curr_approved_idx)
                return a, 0
