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

#class MetaFixedShare(Policy):
#    """
#    Meta fixed share
#    """
#    def __init__(self, alpha: float, eta: float, policy_keys: List[str], policy_dict: Dict[str, Policy], human_max_loss: float):
#        self.policy_keys = policy_keys
#        self.policy_dict = policy_dict
#        self.num_policies = len(policy_dict)
#        self.meta_weights = np.ones(len(policy_dict))
#        self.alpha = min(0.4, alpha)
#        self.eta = eta
#        self.human_max_loss = human_max_loss
#
#    def __str__(self):
#        return "MetaFixedShare"
#
#    def add_expert(self, time_t):
#        for k, policy in self.policy_dict.items():
#            policy.add_expert(time_t)
#
#    def _get_policy_prev_loss(self, time_t: int, model_losses_t: np.ndarray, policy_key: str):
#        policy = self.policy_dict[policy_key]
#        robot_weights, human_weight = policy.get_predict_weights(time_t)
#        policy_loss = np.sum(model_losses_t * robot_weights) + human_weight * self.human_max_loss
#        return policy_loss
#
#    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
#        if indiv_robot_loss_t is None:
#            return
#
#        # Update the meta policy weights first
#        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
#        loss_t = np.array([
#                self._get_policy_prev_loss(time_t - 1, model_losses_t, policy_key) for policy_key in self.policy_keys])
#        print("loss", loss_t)
#
#        update_weight = np.exp(-self.eta * loss_t)
#        self.meta_weights = update_weight * self.meta_weights
#        transition_matrix = np.eye(self.meta_weights.size) * (1 - self.alpha) + np.ones((self.meta_weights.size, self.meta_weights.size)) * self.alpha/self.meta_weights.size
#        self.meta_weights = self.meta_weights.reshape((1,-1)) @ transition_matrix
#        self.meta_weights = self.meta_weights.flatten()
#
#        # Let each policy update their own weights
#        for k, policy in self.policy_dict.items():
#            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights)
#
#
#    def get_predict_weights(self, time_t):
#        denom = np.sum(self.meta_weights)
#        policy_weights = self.meta_weights/denom
#        print("pocliy weights", policy_weights)
#
#        robot_weights = 0
#        human_weight = 0
#        for i, policy_key in enumerate(self.policy_keys):
#            policy = self.policy_dict[policy_key]
#            policy_robot_weights, policy_human_weight = policy.get_predict_weights(time_t)
#            robot_weights += policy_robot_weights * policy_weights[i]
#            human_weight += policy_human_weight * policy_weights[i]
#        return robot_weights, human_weight
#
class MetaExpWeightingSimple(Policy):
    """
    Meta exponential weighting
    """
    def __init__(self, eta, policy_keys: List[str], policy_dict: Dict[str, Policy], human_max_loss: float):
        self.policy_keys = policy_keys
        self.policy_dict = policy_dict
        self.num_policies = len(policy_dict)
        self.meta_weights = np.ones(len(policy_dict))
        self.eta = eta
        self.human_max_loss = human_max_loss

    def __str__(self):
        return "MetaExp"

    def add_expert(self, time_t):
        for k, policy in self.policy_dict.items():
            policy.add_expert(time_t)

    def _get_policy_prev_loss(self, time_t: int, model_losses_t: np.ndarray, policy_key: str):
        policy = self.policy_dict[policy_key]
        robot_weights, human_weight = policy.get_predict_weights(time_t)
        policy_loss = np.sum(model_losses_t * robot_weights) + human_weight * self.human_max_loss
        return policy_loss

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        # Update the meta policy weights first
        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        loss_t = np.array([
                self._get_policy_prev_loss(time_t - 1, model_losses_t, policy_key) for policy_key in self.policy_keys])
        print("loss", loss_t)

        update_weight = np.exp(-self.eta * loss_t)
        self.meta_weights = update_weight * self.meta_weights

        # Let each policy update their own weights
        for k, policy in self.policy_dict.items():
            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights)


    def get_predict_weights(self, time_t):
        denom = np.sum(self.meta_weights)
        policy_weights = self.meta_weights/denom
        print("pocliy weights", policy_weights)

        robot_weights = 0
        human_weight = 0
        for i, policy_key in enumerate(self.policy_keys):
            policy = self.policy_dict[policy_key]
            policy_robot_weights, policy_human_weight = policy.get_predict_weights(time_t)
            robot_weights += policy_robot_weights * policy_weights[i]
            human_weight += policy_human_weight * policy_weights[i]
        return robot_weights, human_weight


class BlindApproval(Policy):
    def __init__(self, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def get_predict_weights(self, time_t: int):
        a = np.zeros(self.curr_num_experts)
        a[-1] = 1
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

class MeanApproval(TTestApproval):
    def __init__(self, num_experts: int, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.curr_num_experts = 0
        self.num_experts = num_experts
        self.loss_histories = [[] for i in range(self.num_experts)]
        self.curr_approved_idx = 0

    def get_predict_weights(self, time_t: int):
        if time_t == 0:
            return np.zeros(1), 1

        best_model_idx = self.curr_approved_idx
        baseline_model_loss = np.mean(self.loss_histories[self.curr_approved_idx][-1])
        best_model_loss = baseline_model_loss

        for i in range(self.curr_approved_idx + 1, self.curr_num_experts - 1):
            new_model_loss = np.mean(self.loss_histories[i][-1])
            if new_model_loss < best_model_loss:
                best_model_idx = i
                best_model_loss = new_model_loss

        if best_model_loss < self.human_max_loss:
            if best_model_idx != self.curr_approved_idx:
                self.curr_approved_idx = best_model_idx
            a = np.zeros(self.curr_num_experts)
            a[best_model_idx] = 1
            return a, 0
        else:
            a = np.zeros(self.curr_num_experts)
            return a, 1

class OptimisticPolicy(Policy):
    def predict_next_losses(self, time_t: int):
        predictions = np.array(
            [
                self.fit_and_predict(self.loss_histories[i])
                for i in range(self.curr_num_experts)
            ]
        )
        return predictions

    def fit_and_predict(self, losses):
        return self.time_trend_predictor.forecast(losses)

class FixedShareWithBlind(Policy):
    """
    Fixed Share
    First expert is human
    Second expert is the blind approval
    """

    def __init__(
            self, num_experts: int, eta: float, human_max_loss: float, alpha: float = 0.1
    ):
        assert eta > 0
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.alpha = alpha
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with human and blind weight
        self.weights = np.array([1, 2])

        self.loss_histories = np.zeros((self.num_experts, 1))
        print("alpha", self.alpha)

    def __str__(self):
        return "FixedShareWithBlind"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        self.weights = np.concatenate([self.weights, [0]])

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        new_losses = np.concatenate(
            [
                model_losses_t,
                [0] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        v_weights = self.weights * np.exp(- self.eta * np.concatenate([
            [
                self.human_max_loss,  # human loss
                model_losses_t[-1]],  # blind loss
            model_losses_t]))
        self.weights = (1 - self.alpha) * v_weights + self.alpha * np.mean(v_weights)
        # Adding normalization to prevent numerical underflow
        #self.weights /= np.max(self.weights)
        print("fixedshare blind wei", self.weights[0], self.weights[1], self.weights[2:].sum())

    def get_predict_weights(self, time_t: int):
        all_weights = self.weights / np.sum(self.weights)
        robot_weights = all_weights[2:]
        print("blind weight", all_weights[1])
        robot_weights[-1] += all_weights[1]
        return robot_weights, all_weights[0]

class OptimisticBaselineMonotonicFixedShare(Policy):
    """
    Fixed Share
    First expert is human
    enforce monotonicity in the expert seq
    """

    def __init__(
            self, num_experts: int, eta: float, human_max_loss: float, alpha: float = 0.1, baseline_alpha: float= 0.01,
    ):
        assert eta > 0
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.alpha = alpha
        self.baseline_alpha = baseline_alpha
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with only human weight
        self.const_baseline_weight = 0.1 #np.exp(-num_experts * self.human_max_loss)
        self.const_baseline_optim_weight = self.const_baseline_weight
        self.baseline_weights = np.ones(1) * (1 - self.const_baseline_weight)
        self.baseline_optim_weights = np.ones(1) * (1 - self.const_baseline_weight)

        self.loss_histories = np.zeros((num_experts, 1))
        print("alpha", self.alpha)

    def __str__(self):
        return "OptimisticBaselineMonotonicFixedShare"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        if self.curr_num_experts == 1:
            self.weights = np.ones(1) * (1 - self.const_baseline_weight)
            self.optim_weights = np.ones(1) * (1 - self.const_baseline_weight)
        else:
            self.baseline_weights = np.concatenate([self.baseline_weights, [0]])
            self.weights = np.concatenate([self.weights, [0]])
            self.baseline_optim_weights = np.concatenate([self.baseline_optim_weights, [0]])
            self.optim_weights = np.concatenate([self.optim_weights, [0]])

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        forecaster_loss = np.sum(
            prev_weights * np.concatenate([[self.human_max_loss], model_losses_t])
        )
        new_losses = np.concatenate(
            [
                model_losses_t,
                [0] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        v_weights = self.weights * np.exp(- self.eta * model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(- self.eta * self.human_max_loss * np.ones(self.baseline_weights.size))

        # heavily weight the new models when transitioning away
        transition_matrix11 = np.eye(self.weights.size) * (1 - self.alpha)
        for i in range(self.weights.size - 1):
            transition_matrix11[i,i] = 1 - self.alpha - self.baseline_alpha
            transition_matrix11[i,i + 1:] = (self.alpha)/(self.weights.size - i - 1)
            #transition_matrix11[i,-1] = self.alpha
        transition_matrix12 = np.eye(self.weights.size, self.baseline_weights.size,k=1) * self.baseline_alpha
        transition_matrix21 = np.zeros((self.baseline_weights.size, self.weights.size))
        transition_matrix21[0,-1] = self.alpha
        for i in range(1,self.weights.size - 1):
            transition_matrix21[i,i + 1:] = (self.alpha)/(self.weights.size - i - 1)
            #transition_matrix21[i,-3:] = (self.alpha)/(3)
            #transition_matrix21[i,-1] = self.alpha
        transition_matrix22 = np.eye(self.baseline_weights.size) * (1 - self.alpha)
        transition_matrix = np.block([[transition_matrix11, transition_matrix12], [transition_matrix21, transition_matrix22]])
        transition_matrix[:,-1] = 1 - transition_matrix[:,:-1].sum(axis=1)

        #print("TRAN", transition_matrix)
        assert not np.any(np.isnan(transition_matrix))
        assert np.all(np.isclose(1, np.sum(transition_matrix, axis=1)))

        combo_vector = np.concatenate([v_weights, v_baseline_weights])

        new_combo_weights = np.matmul(combo_vector.reshape((1, -1)), transition_matrix).flatten()
        self.weights = new_combo_weights[:self.weights.size]
        self.baseline_weights = new_combo_weights[self.weights.size:]
        self.const_baseline_weight = self.const_baseline_weight * np.exp(- self.eta * self.human_max_loss)
        # Adding normalization to prevent numerical underflow
        normalization_factor = np.max(self.weights)
        self.weights /= normalization_factor
        self.baseline_weights /= normalization_factor
        self.const_baseline_weight /= normalization_factor

        # Don't bother using any algorithms we predict to be worse than the human
        self.optim_weights = v_weights * np.exp(- self.eta * model_losses_t) * (model_losses_t < self.human_max_loss)
        self.baseline_optim_weights = v_baseline_weights * np.exp(- self.eta * self.human_max_loss)
        self.const_baseline_optim_weight = self.const_baseline_weight * np.exp(- self.eta * self.human_max_loss)


    def get_predict_weights(self, time_t: int):
        denom = (np.sum(self.optim_weights) + np.sum(self.baseline_optim_weights) + self.const_baseline_optim_weight)
        robot_optim_weights = self.optim_weights / (np.sum(self.optim_weights) + np.sum(self.baseline_optim_weights) + self.const_baseline_optim_weight)
        baseline_weight = 1 - np.sum(robot_optim_weights)
        #print("CONST", self.const_baseline_optim_weight/denom)
        #print("other", np.sum(self.baseline_optim_weights)/denom)
        return robot_optim_weights, baseline_weight

class OptimisticMonotonicFixedShare(Policy):
    """
    Fixed Share
    First expert is human
    enforce monotonicity in the expert seq
    """

    def __init__(
            self, num_experts: int, eta: float, human_max_loss: float, alpha: float = 0.1, baseline_alpha: float= 0.05,
    ):
        assert eta > 0
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.alpha = alpha
        self.baseline_alpha = baseline_alpha
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with only human weight
        self.baseline_weights = np.ones(1)
        self.baseline_optim_weights = np.ones(1)

        self.loss_histories = np.zeros((num_experts, 1))
        print("alpha", self.alpha)

    def __str__(self):
        return "OptimisticMonotonicFixedShare"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        if self.curr_num_experts == 1:
            self.weights = np.ones(1)
            self.optim_weights = np.ones(1)
        else:
            self.baseline_weights = np.concatenate([self.baseline_weights, [0]])
            self.weights = np.concatenate([self.weights, [0]])
            self.baseline_optim_weights = np.concatenate([self.baseline_optim_weights, [0]])
            self.optim_weights = np.concatenate([self.optim_weights, [0]])

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        forecaster_loss = np.sum(
            prev_weights * np.concatenate([[self.human_max_loss], model_losses_t])
        )
        new_losses = np.concatenate(
            [
                model_losses_t,
                [0] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        v_weights = self.weights * np.exp(- self.eta * model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(- self.eta * self.human_max_loss * np.ones(self.baseline_weights.size))

        # heavily weight the new models when transitioning away
        transition_matrix11 = np.eye(self.weights.size) * (1 - self.alpha)
        for i in range(self.weights.size - 1):
            transition_matrix11[i,i] = 1 - self.alpha - self.baseline_alpha
            #transition_matrix11[i,i + 1:] = (self.alpha)/(self.weights.size - i - 1)
            transition_matrix11[i,-1] = self.alpha
        transition_matrix12 = np.eye(self.weights.size, self.baseline_weights.size,k=1) * self.baseline_alpha
        transition_matrix21 = np.zeros((self.baseline_weights.size, self.weights.size))
        transition_matrix21[0,-1] = self.alpha
        for i in range(1,self.weights.size - 1):
            transition_matrix21[i,-1] = self.alpha
        transition_matrix22 = np.eye(self.baseline_weights.size) * (1 - self.alpha)
        transition_matrix = np.block([[transition_matrix11, transition_matrix12], [transition_matrix21, transition_matrix22]])
        transition_matrix[:,-1] = 1 - transition_matrix[:,:-1].sum(axis=1)

        #print("TRAN", transition_matrix)
        assert not np.any(np.isnan(transition_matrix))
        assert np.all(np.isclose(1, np.sum(transition_matrix, axis=1)))

        combo_vector = np.concatenate([v_weights, v_baseline_weights])

        new_combo_weights = np.matmul(combo_vector.reshape((1, -1)), transition_matrix).flatten()
        self.weights = new_combo_weights[:self.weights.size]
        self.baseline_weights = new_combo_weights[self.weights.size:]
        # Adding normalization to prevent numerical underflow
        #self.weights /= np.max(self.weights)
        self.optim_weights = v_weights * np.exp(- self.eta * model_losses_t)
        self.baseline_optim_weights = v_baseline_weights * np.exp(- self.eta * self.human_max_loss)


    def get_predict_weights(self, time_t: int):
        robot_optim_weights = self.optim_weights / (np.sum(self.optim_weights) + np.sum(self.baseline_optim_weights))
        baseline_weight = 1 - np.sum(robot_optim_weights)
        return robot_optim_weights, baseline_weight

class MonotonicBaselineFixedShare(Policy):
    """
    Fixed Share with baseline
    First expert is human
    enforce monotonicity in the expert seq
    """

    def __init__(
            self, num_experts: int, eta: float, human_max_loss: float, alpha: float = 0.1, baseline_alpha: float= 0.01,
    ):
        assert eta > 0
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.alpha = alpha
        self.baseline_alpha = baseline_alpha
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with only human weight
        print("orig", 1/num_experts)
        print("new", np.power(1/num_experts, self.human_max_loss))
        self.const_baseline_weight = np.exp(-num_experts * self.human_max_loss/10)
        #self.const_baseline_weight = 1/num_experts
        print("CONST", self.const_baseline_weight)
        self.baseline_weights = np.ones(1) * (1 - self.const_baseline_weight)

        self.loss_histories = np.zeros((num_experts, 1))
        print("alpha", self.alpha)

    def __str__(self):
        return "FixedShare"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        if self.curr_num_experts == 1:
            self.weights = np.ones(1) * (1 - self.const_baseline_weight)
        else:
            self.baseline_weights = np.concatenate([self.baseline_weights, [0]])
            self.weights = np.concatenate([self.weights, [0]])

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        forecaster_loss = np.sum(
            prev_weights * np.concatenate([[self.human_max_loss], model_losses_t])
        )
        new_losses = np.concatenate(
            [
                model_losses_t,
                [0] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        v_weights = self.weights * np.exp(- self.eta * model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(- self.eta * self.human_max_loss * np.ones(self.baseline_weights.size))

        # heavily weight the new models when transitioning away
        transition_matrix11 = np.eye(self.weights.size) * (1 - self.alpha)
        for i in range(self.weights.size - 1):
            transition_matrix11[i,i] = 1 - self.alpha - self.baseline_alpha
            transition_matrix11[i,i + 1:] = (self.alpha)/(self.weights.size - i - 1)
            #transition_matrix11[i,-1] = self.alpha
        transition_matrix12 = np.eye(self.weights.size, self.baseline_weights.size,k=1) * self.baseline_alpha
        transition_matrix21 = np.zeros((self.baseline_weights.size, self.weights.size))
        transition_matrix21[0,-1] = self.alpha
        for i in range(1,self.weights.size - 1):
            transition_matrix11[i,i + 1:] = (self.alpha)/(self.weights.size - i - 1)
            #transition_matrix21[i,-1] = self.alpha
        transition_matrix22 = np.eye(self.baseline_weights.size) * (1 - self.alpha)
        transition_matrix = np.block([[transition_matrix11, transition_matrix12], [transition_matrix21, transition_matrix22]])
        transition_matrix[:,-1] = 1 - transition_matrix[:,:-1].sum(axis=1)

        #print("TRAN", transition_matrix)
        assert not np.any(np.isnan(transition_matrix))
        assert np.all(np.isclose(1, np.sum(transition_matrix, axis=1)))

        combo_vector = np.concatenate([v_weights, v_baseline_weights])

        new_combo_weights = np.matmul(combo_vector.reshape((1, -1)), transition_matrix).flatten()
        self.weights = new_combo_weights[:self.weights.size]
        self.baseline_weights = new_combo_weights[self.weights.size:]
        self.const_baseline_weight = self.const_baseline_weight * np.exp(- self.eta * self.human_max_loss)
        # Adding normalization to prevent numerical underflow
        #self.weights /= np.max(self.weights)

    def get_predict_weights(self, time_t: int):
        robot_weights = self.weights / (np.sum(self.weights) + np.sum(self.baseline_weights) + self.const_baseline_weight)
        baseline_weight = 1 - np.sum(robot_weights)
        return robot_weights, baseline_weight


class MonotonicFixedShare(Policy):
    """
    Fixed Share
    First expert is human
    enforce monotonicity in the expert seq
    """

    def __init__(
            self, num_experts: int, eta: float, human_max_loss: float, alpha: float = 0.1, baseline_alpha: float= 0.05,
    ):
        assert eta > 0
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.alpha = alpha
        self.baseline_alpha = baseline_alpha
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with only human weight
        self.baseline_weights = np.ones(1)

        self.loss_histories = np.zeros((num_experts, 1))
        print("alpha", self.alpha)

    def __str__(self):
        return "FixedShare"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        if self.curr_num_experts == 1:
            self.weights = np.ones(1)
        else:
            self.baseline_weights = np.concatenate([self.baseline_weights, [0]])
            self.weights = np.concatenate([self.weights, [0]])

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        forecaster_loss = np.sum(
            prev_weights * np.concatenate([[self.human_max_loss], model_losses_t])
        )
        new_losses = np.concatenate(
            [
                model_losses_t,
                [0] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        v_weights = self.weights * np.exp(- self.eta * model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(- self.eta * self.human_max_loss * np.ones(self.baseline_weights.size))

        # even split transition probs across later models
        #transition_matrix11 = np.eye(self.weights.size) * (1 - self.alpha)
        #for i in range(self.weights.size - 1):
        #    transition_matrix11[i,i] = 1 - self.alpha - self.baseline_alpha
        #    transition_matrix11[i,i + 1:] = (self.alpha)/(self.weights.size - i - 1)
        #transition_matrix12 = np.eye(self.weights.size, self.baseline_weights.size,k=1) * self.baseline_alpha
        #transition_matrix21 = np.zeros((self.baseline_weights.size, self.weights.size))
        #for i in range(self.weights.size - 1):
        #    transition_matrix21[i,i + 1:] = (2 * self.alpha)/(self.weights.size - i - 1)
        #transition_matrix22 = np.eye(self.baseline_weights.size) * (1 - self.alpha * 2)
        #transition_matrix = np.block([[transition_matrix11, transition_matrix12], [transition_matrix21, transition_matrix22]])
        #transition_matrix[:,-1] = 1 - transition_matrix[:,:-1].sum(axis=1)

        # heavily weight the new models when transitioning away
        transition_matrix11 = np.eye(self.weights.size) * (1 - self.alpha)
        for i in range(self.weights.size - 1):
            transition_matrix11[i,i] = 1 - self.alpha - self.baseline_alpha
            #transition_matrix11[i,i + 1:] = (self.alpha)/(self.weights.size - i - 1)
            transition_matrix11[i,-1] = self.alpha
        transition_matrix12 = np.eye(self.weights.size, self.baseline_weights.size,k=1) * self.baseline_alpha
        transition_matrix21 = np.zeros((self.baseline_weights.size, self.weights.size))
        transition_matrix21[0,-1] = self.alpha
        for i in range(1,self.weights.size - 1):
            transition_matrix21[i,-1] = self.alpha
        transition_matrix22 = np.eye(self.baseline_weights.size) * (1 - self.alpha)
        transition_matrix = np.block([[transition_matrix11, transition_matrix12], [transition_matrix21, transition_matrix22]])
        transition_matrix[:,-1] = 1 - transition_matrix[:,:-1].sum(axis=1)

        #print("TRAN", transition_matrix)
        assert not np.any(np.isnan(transition_matrix))
        assert np.all(np.isclose(1, np.sum(transition_matrix, axis=1)))

        combo_vector = np.concatenate([v_weights, v_baseline_weights])

        new_combo_weights = np.matmul(combo_vector.reshape((1, -1)), transition_matrix).flatten()
        self.weights = new_combo_weights[:self.weights.size]
        self.baseline_weights = new_combo_weights[self.weights.size:]
        # Adding normalization to prevent numerical underflow
        #self.weights /= np.max(self.weights)

    def get_predict_weights(self, time_t: int):
        robot_weights = self.weights / (np.sum(self.weights) + np.sum(self.baseline_weights))
        baseline_weight = 1 - np.sum(robot_weights)
        return robot_weights, baseline_weight

class FixedShare(Policy):
    """
    Fixed Share
    First expert is human
    """

    def __init__(
            self, num_experts: int, eta: float, human_max_loss: float, alpha: float = 0.1
    ):
        assert eta > 0
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.alpha = alpha
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with only human weight
        self.weights = np.ones(1)

        self.loss_histories = np.zeros((num_experts, 1))
        print("alpha", self.alpha)

    def __str__(self):
        return "FixedShare"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        if self.curr_num_experts == 1:
            self.weights = np.concatenate([self.weights, [2]])
        else:
            self.weights = np.concatenate([self.weights, [0]])

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        forecaster_loss = np.sum(
            prev_weights * np.concatenate([[self.human_max_loss], model_losses_t])
        )
        new_losses = np.concatenate(
            [
                model_losses_t,
                [0] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        v_weights = self.weights * np.exp(- self.eta * np.concatenate([[self.human_max_loss], model_losses_t]))
        self.weights = (1 - self.alpha) * v_weights + self.alpha * np.mean(v_weights)
        # Adding normalization to prevent numerical underflow
        #self.weights /= np.max(self.weights)
        print("self wei", self.weights)

    def get_predict_weights(self, time_t: int):
        all_weights = self.weights / np.sum(self.weights)
        print("weights", all_weights)
        return all_weights[1:], all_weights[0]

class OptimisticMirrorDescent(Policy):
    """
    Optimistic mirror descent, modified to let in new experts
    First expert is human
    """

    def __init__(
        self, num_experts: int, eta: float, human_max_loss: float, time_trend_predictor
    ):
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.num_experts = num_experts
        self.weights = np.ones(num_experts)
        self.curr_num_experts = 0
        self.time_trend_predictor = time_trend_predictor

        self.loss_histories = np.zeros((num_experts, 1))

    def __str__(self):
        return "OMD"

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        forecaster_loss = np.sum(
            prev_weights * np.concatenate([[self.human_max_loss], model_losses_t])
        )
        new_losses = np.concatenate(
            [
                model_losses_t,
                [forecaster_loss] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)

    def get_predict_weights(self, time_t: int):
        predictions = np.array(
            [
                self.fit_and_predict(self.loss_histories[i])
                for i in range(self.curr_num_experts)
            ]
        )

        num_steps = self.loss_histories.shape[1]
        projected_expert_losses = (
            np.sum(self.loss_histories[: self.curr_num_experts], axis=1) + predictions
        )
        projected_human_loss = self.human_max_loss * num_steps
        raw_weights = np.exp(
            -self.eta
            * np.concatenate([[projected_human_loss], projected_expert_losses])
        )
        all_weights = raw_weights / np.sum(raw_weights)
        return all_weights[1:], all_weights[0]

    def predict_next_losses(self, time_t: int):
        predictions = np.array(
            [
                self.fit_and_predict(self.loss_histories[i])
                for i in range(self.curr_num_experts)
            ]
        )
        return predictions

    def fit_and_predict(self, losses):
        return self.time_trend_predictor.forecast(losses)

class MirrorDescent(Policy):
    """
    Ordinary exponential weighting, modified to let in new experts
    First expert is human
    """

    def __init__(self, num_experts: int, eta: float, human_max_loss: float):
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.num_experts = num_experts
        self.weights = np.ones(num_experts)
        self.curr_num_experts = 0

        self.loss_histories = np.zeros((num_experts, 1))

    def __str__(self):
        return "ExpWeighting"

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        forecaster_loss = np.sum(
            prev_weights * np.concatenate([[self.human_max_loss], model_losses_t])
        )
        new_losses = np.concatenate(
            [
                model_losses_t,
                [forecaster_loss] * (self.num_experts - self.curr_num_experts),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)

    def get_predict_weights(self, time_t):
        expert_losses = np.sum(self.loss_histories[: self.curr_num_experts], axis=1)
        human_loss = self.human_max_loss * (self.loss_histories.shape[1] - 1)
        raw_weights = np.exp(-self.eta * np.concatenate([[human_loss], expert_losses]))
        all_weights = raw_weights / np.sum(raw_weights)
        return all_weights[1:], all_weights[0]

#class OptimisticPolicy(Policy):
#    """
#    This just picks the expert with the smallest predicted loss.
#    No exponential weighting. No guarantees
#    """
#
#    def __init__(
#        self, num_experts: int, eta: float, human_max_loss: float, time_trend_predictor
#    ):
#        self.human_max_loss = human_max_loss
#        self.eta = eta
#        self.num_experts = num_experts
#        self.weights = np.ones(num_experts)
#        self.curr_num_experts = 0
#        self.time_trend_predictor = time_trend_predictor
#
#        self.loss_histories = []
#
#    def __str__(self):
#        return "Optimistic"
#
#    def add_expert(self, time_t):
#        self.curr_num_experts += 1
#        self.loss_histories.append([])
#
#    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
#        if indiv_robot_loss_t is None:
#            return
#
#        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
#        for i in range(self.curr_num_experts):
#            self.loss_histories[i].append(model_losses_t[i])
#
#    def get_predict_weights(self, time_t: int):
#        predictions = np.array(
#            [
#                self.fit_and_predict(np.array(self.loss_histories[i]))
#                for i in range(self.curr_num_experts)
#            ]
#        )
#
#        all_pred = np.concatenate([[self.human_max_loss], predictions])
#        minimizer = np.argmin(all_pred)
#        all_weights = np.zeros(all_pred.size)
#        all_weights[minimizer] = 1
#
#        return all_weights[1:], all_weights[0]
#
#    def predict_next_losses(self, time_t: int):
#        predictions = np.array(
#            [
#                self.fit_and_predict(np.array(self.loss_histories[i]))
#                for i in range(self.curr_num_experts)
#            ]
#        )
#        return predictions
#
#    def fit_and_predict(self, losses):
#        return self.time_trend_predictor.forecast(losses)
