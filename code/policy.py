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

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        return

    def predict_next_losses(self, time_t: int):
        return np.zeros(self.curr_num_experts)

class BaselinePolicy(Policy):
    def get_predict_weights(self, time_t: int):
        return np.zeros(self.curr_num_experts), 1

class BlindApproval(Policy):
    def __init__(self, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def get_predict_weights(self, time_t: int):
        a = np.zeros(self.curr_num_experts)
        a[-1] = 1
        print(time_t, "chosen robot", np.where(a))
        return a, 0

class TTestApproval(Policy):
    def __init__(self, num_experts: int, human_max_loss: float, factor: float = 1.96):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0
        self.num_experts = num_experts
        self.loss_histories = [[] for i in range(self.num_experts)]
        self.curr_approved_idx = 0
        self.factor = factor

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        for i in range(self.curr_num_experts):
            self.loss_histories[i].append(indiv_robot_loss_t[i,:])
        for i in range(self.curr_num_experts, self.num_experts):
            self.loss_histories[i].append([])

    def get_predict_weights(self, time_t: int):
        best_model_idx = self.curr_approved_idx
        best_upper_ci = 0
        differences = []
        for i in range(self.curr_approved_idx + 1, self.curr_num_experts - 1):
            new_model_loss = np.concatenate(self.loss_histories[i][i:])
            baseline_model_loss = np.concatenate(self.loss_histories[self.curr_approved_idx][i:])
            loss_improvement = new_model_loss - baseline_model_loss
            mean_improve = np.mean(loss_improvement)
            differences.append(mean_improve)
            upper_ci = mean_improve + self.factor * np.sqrt(np.var(loss_improvement)/new_model_loss.size)
            is_better = upper_ci < 0
            if is_better and upper_ci < best_upper_ci:
                best_model_idx = i
                best_upper_ci = upper_ci

        self.curr_approved_idx = best_model_idx

        a = np.zeros(self.curr_num_experts)
        a[best_model_idx] = 1
        return a, 0

