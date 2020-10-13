from itertools import product
from scipy import special
import numpy as np
from policy import Policy


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
        const_baseline_weight: float = 1e-10,
        pred_t_factor: float = 1,
        num_back_batches: int = 3,
    ):
        self.human_max_loss = human_max_loss

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

        self.pred_t_factor = pred_t_factor
        self.batch_sizes = []
        # print("self.const_baseline_weight", self.const_baseline_weight)

        self.loss_histories = np.zeros((num_experts, 1))
        self.var_loss_histories = np.zeros((num_experts, 1))

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

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is None:
            return

        model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
        var_model_losses_t = np.var(indiv_robot_loss_t, axis=1)
        if self.curr_num_experts > model_losses_t.size:
            model_losses_t = np.concatenate([model_losses_t, [0]])
            var_model_losses_t = np.concatenate([var_model_losses_t, [0]])
        new_losses = np.concatenate(
            [
                model_losses_t,
                [0] * (self.num_experts - model_losses_t.size),
            ]
        ).reshape((-1, 1))
        var_new_losses = np.concatenate(
            [
                var_model_losses_t,
                [0] * (self.num_experts - model_losses_t.size),
            ]
        ).reshape((-1, 1))
        self.loss_histories = np.concatenate([self.loss_histories, new_losses], axis=1)
        self.var_loss_histories = np.concatenate(
            [self.var_loss_histories, var_new_losses], axis=1
        )
        self.batch_sizes.append(indiv_robot_loss_t.shape[1])
        v_weights = self.weights * np.exp(-self.etas[0] * model_losses_t)
        # print(self.weights, model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(
            -self.etas[0] * self.human_max_loss * np.ones(self.baseline_weights.size)
        )

        # heavily weight the new models when transitioning away
        transition_matrix11 = np.eye(self.weights.size) * (1 - self.alpha)
        for i in range(self.weights.size - 1):
            transition_matrix11[i, i] = 1 - self.alpha - self.baseline_alpha
            transition_matrix11[i, i + 1 :] = (self.alpha) / (self.weights.size - i - 1)
            # transition_matrix11[i,-1] = self.alpha
        transition_matrix12 = (
            np.eye(self.weights.size, self.baseline_weights.size, k=1)
            * self.baseline_alpha
        )
        transition_matrix21 = np.zeros((self.baseline_weights.size, self.weights.size))
        transition_matrix21[0, -1] = self.alpha
        for i in range(1, self.weights.size - 1):
            transition_matrix21[i, i + 1 :] = (self.alpha) / (self.weights.size - i - 1)
            # transition_matrix21[i,-3:] = (self.alpha)/(3)
            # transition_matrix21[i,-1] = self.alpha
        transition_matrix22 = np.eye(self.baseline_weights.size) * (1 - self.alpha)
        transition_matrix = np.block(
            [
                [transition_matrix11, transition_matrix12],
                [transition_matrix21, transition_matrix22],
            ]
        )
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

        # TODO: fix up the standard error estimate
        predictions = np.mean(
            self.loss_histories[: time_t + 1, -self.num_back_batches :], axis=1
        ) + self.pred_t_factor * np.sqrt(
            np.mean(
                self.var_loss_histories[: time_t + 1, -self.num_back_batches :], axis=1
            )
            / np.sum(self.batch_sizes[-self.num_back_batches :])
        )
        predictions[-1] = self.human_max_loss * 2
        log_model_update_factors = -self.etas[1] * predictions
        log_baseline_update_factor = -self.etas[1] * self.human_max_loss

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

        self.optim_weights *= predictions <= self.human_max_loss

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
    ):
        self.eta = eta
        self.eta_list = eta_list
        self.eta_list_size = len(eta_list)

        self.eta_indexes = np.arange(len(eta_list))
        print("ETA INDEX", self.eta_indexes)

        self.policy_dict = {}
        for idx, etas in enumerate(eta_list):
            self.policy_dict[etas] = ValidationPolicy(
                num_experts, np.array(etas), human_max_loss
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
        self, time_t: int, model_losses_t: np.ndarray, policy: Policy
    ):
        robot_weights, human_weight = policy.weight_history[time_t]
        assert np.isclose(robot_weights.sum() + human_weight, 1)
        #print(robot_weights.size, model_losses_t.size)
        policy_loss = (
                np.sum(model_losses_t[:robot_weights.size] * robot_weights) + human_weight * self.human_max_loss
        )
        return policy_loss

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is not None:
            # Update the meta policy weights first
            model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
            loss_t = np.zeros(self.eta_list_size)
            for idx, etas in enumerate(self.eta_list):
                loss_t[idx] = self._get_policy_prev_loss(
                    time_t - 1, model_losses_t, self.policy_dict[etas]
                )
                # print("policy loss", etas, loss_t[idx])
            self.loss_ts += loss_t
            self.meta_weights = self.meta_weights * np.exp(-self.eta * loss_t)

        # Let each policy update their own weights
        for policy in self.policy_dict.values():
            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights=None)

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
            print(
                "policy",
                policy_eta,
                policy_weight,
                policy_human_weight,
                np.argmax(policy_robot_weights),
            )
        print("ETAS", biggest_eta, biggest_weight)
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


class MetaGridSearch(MetaExpWeightingList):
    """
    Meta grid search
    """

    def _get_policy_etas(self, etas):
        etas = list(etas)
        max_baseline = np.max(self.eta_grid[-1])
        etas[3] = max_baseline - etas[3]
        # make sure alphas at most sum to 1
        if etas[2] + etas[3] >= 1:
            return np.array([etas[0], etas[1], etas[2], 0])
        else:
            return np.array(etas)

    def __init__(self, eta: float, eta_grid, num_experts: int, human_max_loss: float):
        self.eta = eta
        self.eta_grid = eta_grid

        self.eta_indexes = [list(range(s.size)) for s in self.eta_grid]
        print("ETA INDEX", self.eta_indexes)

        self.regularization = np.zeros([s.size for s in self.eta_grid])
        self.policy_dict = {}
        for indexes in product(*self.eta_indexes):
            etas = [v[i] for i, v in zip(indexes, self.eta_grid)]
            policy_etas = self._get_policy_etas(etas)
            self.policy_dict[tuple(etas)] = ValidationPolicy(
                num_experts, policy_etas, human_max_loss
            )
            self.regularization[indexes] = 0 #.001 * np.sum(
            #    np.power(np.array(policy_etas), 2)
            #)

        self.loss_ts = np.zeros([s.size for s in self.eta_grid])
        self.human_max_loss = human_max_loss
        self.human_min_reward = 1 - human_max_loss
        self.num_experts = num_experts

        self.meta_weights = np.ones([s.size for s in self.eta_grid])

    def __str__(self):
        return "MetaGridSearch"

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is not None:
            # Update the meta policy weights first
            model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
            loss_t = np.zeros([s.size for s in self.eta_grid])
            for indexes in product(*self.eta_indexes):
                etas = tuple([etas[i] for i, etas in zip(indexes, self.eta_grid)])
                loss_t[indexes] = self._get_policy_prev_loss(
                    time_t - 1, model_losses_t, self.policy_dict[etas]
                )
            self.loss_ts += loss_t
            regularized_loss = self.eta * self.loss_ts + self.regularization
            best_ind = np.unravel_index(
                np.argmin(regularized_loss, axis=None), regularized_loss.shape
            )
            self.meta_weights[:] = 0
            self.meta_weights[best_ind] = 1

        # Let each policy update their own weights
        for policy in self.policy_dict.values():
            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights)

    def get_predict_weights(self, time_t):
        denom = np.sum(self.meta_weights)
        policy_weights = self.meta_weights / denom

        robot_weights = 0
        human_weight = 0
        biggest_weight = 0
        biggest_eta = None
        for indexes in product(*self.eta_indexes):
            etas = tuple([etas[i] for i, etas in zip(indexes, self.eta_grid)])
            policy_etas = self._get_policy_etas(etas)
            policy = self.policy_dict[etas]
            policy_weight = policy_weights[indexes]
            #if policy_weight == 0:
            #    continue
            policy_robot_weights, policy_human_weight = policy.get_predict_weights(
                time_t
            )
            robot_weights += policy_robot_weights * policy_weight
            human_weight += policy_human_weight * policy_weight
            #print("etas", policy_etas)
            #print(policy_robot_weights)
            assert np.all(policy_robot_weights >= 0)
            assert policy_human_weight >= 0 or np.isclose(policy_human_weight, 0)
            if biggest_weight < policy_weights[indexes]:
                biggest_weight = policy_weights[indexes]
                biggest_eta = policy_etas
            # print("policy", etas, policy_weight, policy_human_weight, np.argmax(policy_robot_weights))
        print("ETAS", biggest_eta, biggest_weight)
        print(
            "time",
            time_t,
            "max weigth robot",
            np.argmax(robot_weights),
            robot_weights.max(),
            "human",
            human_weight,
        )
        assert np.all(robot_weights >= 0)
        assert human_weight > 0 or np.isclose(human_weight, 0)
        return robot_weights, human_weight


# class OptimMetaExpWeightingList(Policy):
#    """
#    optimistic Meta exponential weighting list
#    """
#
#    def _get_policy_etas(self, etas):
#        return np.array(etas)
#
#    def __init__(
#        self,
#        eta: float,
#        eta_list,
#        meta_weights: np.ndarray,
#        num_experts: int,
#        human_max_loss: float,
#    ):
#        self.eta = eta
#        self.eta_list = eta_list
#        self.eta_list_size = len(eta_list)
#
#        self.eta_indexes = np.arange(len(eta_list))
#        print("ETA INDEX", self.eta_indexes)
#
#        self.policy_dict = {}
#        for idx, etas in enumerate(eta_list):
#            self.policy_dict[etas] = ValidationPolicy(
#                num_experts, np.array(etas), human_max_loss
#            )
#
#        self.loss_ts = np.zeros(len(eta_list))
#        self.prev_losses = self.loss_ts
#        self.human_max_loss = human_max_loss
#        self.num_experts = num_experts
#
#        self.meta_weights = meta_weights
#        self.optim_meta_weights = self.meta_weights
#
#    def __str__(self):
#        return "OptimMetaExp"
#
#    def add_expert(self, time_t):
#        for k, policy in self.policy_dict.items():
#            policy.add_expert(time_t)
#
#    def _get_policy_prev_loss(
#        self, time_t: int, model_losses_t: np.ndarray, policy: Policy
#    ):
#        robot_weights, human_weight = policy.weight_history[time_t]
#        assert np.isclose(robot_weights.sum() + human_weight, 1)
#        policy_loss = (
#            np.sum(model_losses_t * robot_weights)
#            + human_weight * self.human_max_loss
#        )
#        return policy_loss
#
#    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
#        if indiv_robot_loss_t is not None:
#            # Update the meta policy weights first
#            model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
#            loss_t = np.zeros(self.eta_list_size)
#            for idx, etas in enumerate(self.eta_list):
#                loss_t[idx] = self._get_policy_prev_loss(
#                    time_t - 1, model_losses_t, self.policy_dict[etas]
#                )
#                print("policy loss", etas, loss_t[idx])
#            self.loss_ts += loss_t
#            self.meta_weights = self.meta_weights * np.exp(-self.eta * loss_t)
#            predictions = loss_t
#            self.optim_meta_weights = self.meta_weights * np.exp(-self.eta * predictions)
#
#        # Let each policy update their own weights
#        for policy in self.policy_dict.values():
#            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights=None)
#
#    def get_predict_weights(self, time_t):
#        denom = np.sum(self.optim_meta_weights)
#        policy_weights = self.optim_meta_weights / denom
#
#        robot_weights = 0
#        human_weight = 0
#        biggest_weight = 0
#        biggest_eta = None
#        for idx, etas in enumerate(self.eta_list):
#            policy = self.policy_dict[etas]
#            policy_eta = self._get_policy_etas(etas)
#            policy_weight = policy_weights[idx]
#            policy_robot_weights, policy_human_weight = policy.get_predict_weights(
#                time_t
#            )
#            robot_weights += policy_robot_weights * policy_weight
#            human_weight += policy_human_weight * policy_weight
#            if biggest_weight < policy_weights[idx]:
#                biggest_weight = policy_weights[idx]
#                biggest_eta = policy_eta
#            print(
#                "policy",
#                policy_eta,
#                policy_weight,
#                policy_human_weight,
#                np.argmax(policy_robot_weights),
#            )
#        print("ETAS", biggest_eta, biggest_weight)
#        print(
#            "time",
#            time_t,
#            "max weigth robot",
#            np.argmax(robot_weights),
#            robot_weights.max(),
#            "human",
#            human_weight,
#        )
#        return robot_weights, human_weight
#
# class OptimMetaFixedShareList(Policy):
#    """
#    optimistic Meta fixed share
#    """
#
#    def _get_policy_etas(self, etas):
#        return np.array(etas)
#
#    def __init__(
#        self,
#        eta: float,
#        eta_list,
#        meta_weights: np.ndarray,
#        num_experts: int,
#        human_max_loss: float,
#    ):
#        self.eta_list = eta_list
#        self.eta_list_size = len(eta_list)
#
#        self.eta = eta
#        self.alpha = 0.05
#        self.transition_matrix = np.eye(self.eta_list_size) * (1 - self.alpha - self.alpha/(self.eta_list_size - 1)) + np.ones((self.eta_list_size, self.eta_list_size)) * self.alpha/(self.eta_list_size - 1)
#
#        self.eta_indexes = np.arange(len(eta_list))
#        print("ETA INDEX", self.eta_indexes)
#
#        self.policy_dict = {}
#        for idx, etas in enumerate(eta_list):
#            self.policy_dict[etas] = ValidationPolicy(
#                num_experts, np.array(etas), human_max_loss
#            )
#
#        self.loss_ts = np.zeros(len(eta_list))
#        self.prev_losses = self.loss_ts
#        self.human_max_loss = human_max_loss
#        self.num_experts = num_experts
#
#        self.meta_weights = meta_weights
#        self.optim_meta_weights = self.meta_weights
#
#    def __str__(self):
#        return "OptimFixedShare"
#
#    def add_expert(self, time_t):
#        for k, policy in self.policy_dict.items():
#            policy.add_expert(time_t)
#
#    def _get_policy_prev_loss(
#        self, time_t: int, model_losses_t: np.ndarray, policy: Policy
#    ):
#        robot_weights, human_weight = policy.weight_history[time_t]
#        assert np.isclose(robot_weights.sum() + human_weight, 1)
#        policy_loss = (
#            np.sum(model_losses_t * robot_weights)
#            + human_weight * self.human_max_loss
#        )
#        return policy_loss
#
#    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
#        if indiv_robot_loss_t is not None:
#            # Update the meta policy weights first
#            model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
#            loss_t = np.zeros(self.eta_list_size)
#            for idx, etas in enumerate(self.eta_list):
#                loss_t[idx] = self._get_policy_prev_loss(
#                    time_t - 1, model_losses_t, self.policy_dict[etas]
#                )
#                print("policy loss", etas, loss_t[idx])
#            self.loss_ts += loss_t
#            self.meta_weights = self.meta_weights * np.exp(-self.eta * loss_t)
#            self.meta_weights = np.matmul(self.meta_weights.reshape((1,-1)), self.transition_matrix).flatten()
#
#            predictions = loss_t
#            self.optim_meta_weights = self.meta_weights * np.exp(-self.eta * predictions)
#
#        # Let each policy update their own weights
#        for policy in self.policy_dict.values():
#            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights=None)
#
#    def get_predict_weights(self, time_t):
#        denom = np.sum(self.optim_meta_weights)
#        policy_weights = self.optim_meta_weights / denom
#
#        robot_weights = 0
#        human_weight = 0
#        biggest_weight = 0
#        biggest_eta = None
#        for idx, etas in enumerate(self.eta_list):
#            policy = self.policy_dict[etas]
#            policy_eta = self._get_policy_etas(etas)
#            policy_weight = policy_weights[idx]
#            policy_robot_weights, policy_human_weight = policy.get_predict_weights(
#                time_t
#            )
#            robot_weights += policy_robot_weights * policy_weight
#            human_weight += policy_human_weight * policy_weight
#            if biggest_weight < policy_weights[idx]:
#                biggest_weight = policy_weights[idx]
#                biggest_eta = policy_eta
#            print(
#                "policy",
#                policy_eta,
#                policy_weight,
#                policy_human_weight,
#                np.argmax(policy_robot_weights),
#            )
#        print("ETAS", biggest_eta, biggest_weight)
#        print(
#            "time",
#            time_t,
#            "max weigth robot",
#            np.argmax(robot_weights),
#            robot_weights.max(),
#            "human",
#            human_weight,
#        )
#        return robot_weights, human_weight
