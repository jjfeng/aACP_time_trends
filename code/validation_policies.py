from itertools import product
import numpy as np
from policy import Policy

class ValidationPolicy(Policy):
    """
    Fixed Share
    First expert is human
    enforce monotonicity in the expert seq
    """

    def __init__(
            self, num_experts: int, etas: np.ndarray, human_max_loss: float, alpha: float = 0.1, baseline_alpha: float= 0.01,
    ):
        self.human_max_loss = human_max_loss
        self.etas = etas
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

    def __str__(self):
        return "ValidationPolicy"

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
        v_weights = self.weights * np.exp(- self.etas[0] * model_losses_t)
        #print(self.weights, model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(- self.etas[0] * self.human_max_loss * np.ones(self.baseline_weights.size))

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
        #print("after transition", self.weights)
        self.baseline_weights = new_combo_weights[self.weights.size:]
        self.const_baseline_weight = self.const_baseline_weight * np.exp(- self.etas[0] * self.human_max_loss)
        # Adding normalization to prevent numerical underflow
        normalization_factor = np.max(self.weights)
        self.weights /= normalization_factor
        self.baseline_weights /= normalization_factor
        self.const_baseline_weight /= normalization_factor

        # Don't bother using any algorithms we predict to be worse than the human
        #self.optim_weights = self.weights * np.exp(- self.eta2 * model_losses_t - self.eta3 * 1.0) #* (model_losses_t < self.human_max_loss)
        #self.baseline_optim_weights = self.baseline_weights * np.exp(- self.eta2 * self.human_max_loss - self.eta3 * self.human_max_loss)
        #self.const_baseline_optim_weight = self.const_baseline_weight * np.exp(- self.eta2 * self.human_max_loss - self.eta3 * self.human_max_loss)

        # Predictors - very optimistic
        blind_losses = np.ones(model_losses_t.shape)
        blind_losses[-1:] = 0

        # Predictors - very pessimistic
        baseline_losses = np.ones(model_losses_t.shape)

        # Predictors - mean approval
        mean_losses = model_losses_t

        all_predictors = [blind_losses, baseline_losses, mean_losses]

        model_update_factors = 1
        baseline_update_factor = 1
        for eta, predictions in zip(self.etas[1:], all_predictors):
            model_update_factors *= np.exp(-eta * predictions)
            baseline_update_factor *= np.exp(-eta * self.human_max_loss)

        self.optim_weights = self.weights * model_update_factors
        self.baseline_optim_weights = self.baseline_weights * baseline_update_factor
        self.const_baseline_optim_weight = self.const_baseline_weight * baseline_update_factor

    def get_predict_weights(self, time_t: int):
        denom = (np.sum(self.optim_weights) + np.sum(self.baseline_optim_weights) + self.const_baseline_optim_weight)
        robot_optim_weights = self.optim_weights / (np.sum(self.optim_weights) + np.sum(self.baseline_optim_weights) + self.const_baseline_optim_weight)
        baseline_weight = 1 - np.sum(robot_optim_weights)
        return robot_optim_weights, baseline_weight

class MetaExpWeighting(Policy):
    """
    Meta exponential weighting
    """
    def __init__(self, eta: float, alpha: float, eta_grid, num_experts: int, human_max_loss: float):
        self.eta = eta
        self.eta_grid = eta_grid

        self.eta_indexes = [list(range(s.size)) for s in self.eta_grid]
        print("ETA INDEX", self.eta_indexes)

        self.policy_dict = {}
        #for etas in product(self.eta_grid):
        for indexes in product(*self.eta_indexes):
            etas = tuple([etas[i] for i, etas in zip(indexes, self.eta_grid)])
            self.policy_dict[etas] = ValidationPolicy(num_experts, np.array(etas), human_max_loss, alpha=alpha)

        self.loss_ts = np.zeros([s.size for s in self.eta_grid])
        self.human_max_loss = human_max_loss
        self.num_experts = num_experts

        self.meta_weights = np.ones([s.size for s in self.eta_grid])

    def __str__(self):
        return "MetaExp"

    def add_expert(self, time_t):
        for k, policy in self.policy_dict.items():
            policy.add_expert(time_t)

    def _get_policy_prev_loss(self, time_t: int, model_losses_t: np.ndarray, policy: Policy):
        robot_weights, human_weight = policy.get_predict_weights(time_t)
        policy_loss = np.sum(model_losses_t * robot_weights) + human_weight * self.human_max_loss
        return policy_loss


    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is not None:
            # Update the meta policy weights first
            model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
            loss_t = np.zeros([s.size for s in self.eta_grid])
            for indexes in product(*self.eta_indexes):
                etas = tuple([etas[i] for i, etas in zip(indexes, self.eta_grid)])
                loss_t[indexes] = self._get_policy_prev_loss(time_t - 1, model_losses_t, self.policy_dict[etas])
            self.loss_ts += loss_t
            self.meta_weights = self.meta_weights * np.exp(-self.eta * loss_t)

        # Let each policy update their own weights
        for policy in self.policy_dict.values():
            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights)

    def get_predict_weights(self, time_t):
        denom = np.sum(self.meta_weights)
        policy_weights = self.meta_weights/denom
        #print(policy_weights)

        robot_weights = 0
        human_weight = 0
        biggest_weight = 0
        biggest_eta = None
        for indexes in product(*self.eta_indexes):
            etas = tuple([etas[i] for i, etas in zip(indexes, self.eta_grid)])
            policy = self.policy_dict[etas]
            policy_weight = policy_weights[indexes]
            policy_robot_weights, policy_human_weight = policy.get_predict_weights(time_t)
            #print(time_t, policy_robot_weights, policy_human_weight)
            robot_weights += policy_robot_weights * policy_weight
            human_weight += policy_human_weight * policy_weight
            #if np.isclose(biggest_weight, policy_weight):
            #    print("close", etas)
            if biggest_weight < policy_weights[indexes]:
                biggest_weight = policy_weights[indexes]
                biggest_eta = etas
        print("ETAS", biggest_eta, biggest_weight)
        print("time", time_t, "best robot", np.argmax(robot_weights), "human", human_weight)
        return robot_weights, human_weight


class MetaGridSearch(MetaExpWeighting):
    """
    Meta grid search
    """
    def __init__(self, eta: float, alpha: float, eta_grid, num_experts: int, human_max_loss: float):
        self.eta = eta
        self.eta_grid = eta_grid

        self.eta_indexes = [list(range(s.size)) for s in self.eta_grid]
        print("ETA INDEX", self.eta_indexes)

        self.regularization = np.zeros([s.size for s in self.eta_grid])
        self.policy_dict = {}
        #for etas in product(self.eta_grid):
        for indexes in product(*self.eta_indexes):
            etas = [v[i] for i, v in zip(indexes, self.eta_grid)]
            policy_etas = np.array([v if i != 2 else (np.max(self.eta_grid[i]) - v) for i, v in enumerate(etas)])
            self.policy_dict[tuple(etas)] = ValidationPolicy(num_experts, policy_etas, human_max_loss, alpha=alpha)
            self.regularization[indexes] = 0.001 * np.sum(np.power(np.array(etas), 2))

        self.loss_ts = np.zeros([s.size for s in self.eta_grid])
        self.human_max_loss = human_max_loss
        self.human_min_reward = 1-human_max_loss
        self.num_experts = num_experts

        self.meta_weights = np.ones([s.size for s in self.eta_grid])

    def __str__(self):
        return "MetaGrid"

    #def _get_policy_prev_loss(self, time_t: int, softmax_rewards_t: np.ndarray, softmax_human: float, policy: Policy):
    #    robot_weights, human_weight = policy.get_predict_weights(time_t)
    #    non_zero = robot_weights > 0
    #    policy_loss = -np.sum(softmax_rewards_t[non_zero] * np.log(robot_weights[non_zero])) - np.log(human_weight) * softmax_human
    #    return policy_loss

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is not None:
            # Update the meta policy weights first
            model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
            #softmax_denom = np.sum(np.exp(self.eta * model_rewards_t)) + np.exp(self.eta * self.human_min_reward)
            #softmax_rewards_t = np.exp(self.eta * model_rewards_t)/softmax_denom
            #softmax_human = np.exp(self.eta * self.human_min_reward)/softmax_denom
            #print(softmax_human)
            #print(self.human_max_loss, 1 - model_rewards_t)
            #softmax_rewards_t = self.eta * model_rewards_t
            #softmax_human = self.eta * self.human_min_reward
            #print("softmax", softmax_rewards_t)
            loss_t = np.zeros([s.size for s in self.eta_grid])
            for indexes in product(*self.eta_indexes):
                etas = tuple([etas[i] for i, etas in zip(indexes, self.eta_grid)])
                loss_t[indexes] = self._get_policy_prev_loss(time_t - 1, model_losses_t, self.policy_dict[etas])
            self.loss_ts += loss_t
            regularized_loss = self.eta * self.loss_ts + self.regularization
            print(np.max(self.regularization), np.min(self.loss_ts))
            best_ind = np.unravel_index(np.argmin(regularized_loss, axis=None), regularized_loss.shape)
            self.meta_weights[:] = 0
            self.meta_weights[best_ind] = 1

        # Let each policy update their own weights
        for policy in self.policy_dict.values():
            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights)

