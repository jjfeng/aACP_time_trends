import numpy as np
from policy import Policy

class ValidationPolicy(Policy):
    """
    Fixed Share
    First expert is human
    enforce monotonicity in the expert seq
    """

    def __init__(
            self, num_experts: int, eta1: float, eta2: float, eta3: float, human_max_loss: float, alpha: float = 0.1, baseline_alpha: float= 0.01,
    ):
        self.human_max_loss = human_max_loss
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
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
        v_weights = self.weights * np.exp(- self.eta1 * model_losses_t)
        #print(self.weights, model_losses_t)
        v_baseline_weights = self.baseline_weights * np.exp(- self.eta1 * self.human_max_loss * np.ones(self.baseline_weights.size))

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
        self.const_baseline_weight = self.const_baseline_weight * np.exp(- self.eta1 * self.human_max_loss)
        # Adding normalization to prevent numerical underflow
        normalization_factor = np.max(self.weights)
        self.weights /= normalization_factor
        self.baseline_weights /= normalization_factor
        self.const_baseline_weight /= normalization_factor

        # Don't bother using any algorithms we predict to be worse than the human
        self.optim_weights = self.weights * np.exp(- self.eta2 * model_losses_t - self.eta3 * 1.0) #* (model_losses_t < self.human_max_loss)
        self.baseline_optim_weights = self.baseline_weights * np.exp(- self.eta2 * self.human_max_loss - self.eta3 * self.human_max_loss)
        self.const_baseline_optim_weight = self.const_baseline_weight * np.exp(- self.eta2 * self.human_max_loss - self.eta3 * self.human_max_loss)


    def get_predict_weights(self, time_t: int):
        denom = (np.sum(self.optim_weights) + np.sum(self.baseline_optim_weights) + self.const_baseline_optim_weight)
        robot_optim_weights = self.optim_weights / (np.sum(self.optim_weights) + np.sum(self.baseline_optim_weights) + self.const_baseline_optim_weight)
        baseline_weight = 1 - np.sum(robot_optim_weights)
        #print("CONST", self.const_baseline_optim_weight/denom)
        #print("other", np.sum(self.baseline_optim_weights)/denom)
        return robot_optim_weights, baseline_weight

class MetaGridSearch(Policy):
    """
    Meta grid search

    eta1: empirical loss
    eta2: prediction for future
    eta3: smaller eta3 means stick to just baseline
    """
    def __init__(self, eta: float, alpha: float, eta1s: np.ndarray, eta2s: np.ndarray, eta3s: np.ndarray, num_experts: int, human_max_loss: float):
        self.eta = eta
        self.eta1s = eta1s
        self.eta2s = eta2s
        self.eta3s = eta3s

        self.regularization = np.zeros((self.eta1s.size, self.eta2s.size, self.eta3s.size))
        self.policy_dict = {}
        for i, eta1 in enumerate(self.eta1s):
            for j, eta2 in enumerate(self.eta2s):
                for k, eta3 in enumerate(self.eta3s):
                    self.policy_dict[(eta1, eta2, eta3)] = ValidationPolicy(num_experts, eta1, eta2, np.max(self.eta3s) - eta3, human_max_loss, alpha=alpha)
                    self.regularization[i, j, k] = 0.01 * np.sum(np.power(np.array([eta1, eta2, eta3]), 2))

        self.loss_ts = np.zeros((self.eta1s.size, self.eta2s.size, self.eta3s.size))
        self.human_max_loss = human_max_loss
        self.num_experts = num_experts

        self.meta_weights = np.zeros((self.eta1s.size, self.eta2s.size, self.eta3s.size))
        self.meta_weights[0,0] = 1

    def __str__(self):
        return "MetaGridSearch"

    def add_expert(self, time_t):
        for policy in self.policy_dict.values():
            policy.add_expert(time_t)

    def _get_policy_prev_loss(self, time_t: int, model_losses_t: np.ndarray, policy: Policy):
        robot_weights, human_weight = policy.get_predict_weights(time_t)
        policy_loss = np.sum(model_losses_t * robot_weights) + human_weight * self.human_max_loss
        return policy_loss

    def update_weights(self, time_t, indiv_robot_loss_t=None, prev_weights=None):
        if indiv_robot_loss_t is not None:
            # Update the meta policy weights first
            model_losses_t = np.mean(indiv_robot_loss_t, axis=1)
            loss_t = np.zeros((self.eta1s.size, self.eta2s.size, self.eta3s.size))
            for i, eta1 in enumerate(self.eta1s):
                for j, eta2 in enumerate(self.eta2s):
                    for k, eta3 in enumerate(self.eta3s):
                        loss_t[i,j, k] = self._get_policy_prev_loss(time_t - 1, model_losses_t, self.policy_dict[(eta1, eta2, eta3)])
            self.loss_ts += loss_t
            regularized_loss = self.eta * self.loss_ts + self.regularization

            best_ind = np.unravel_index(np.argmin(regularized_loss, axis=None), regularized_loss.shape)
            print("BEST ETA", time_t, self.eta1s[best_ind[0]], self.eta2s[best_ind[1]], self.eta3s[best_ind[2]])
            self.meta_weights = np.zeros((self.eta1s.size, self.eta2s.size, self.eta3s.size))
            self.meta_weights[best_ind[0], best_ind[1], best_ind[2]] = 1

        # Let each policy update their own weights
        for k, policy in self.policy_dict.items():
            policy.update_weights(time_t, indiv_robot_loss_t, prev_weights)

    def get_predict_weights(self, time_t):
        denom = np.sum(self.meta_weights)
        policy_weights = self.meta_weights/denom

        robot_weights = 0
        human_weight = 0
        for i, eta1 in enumerate(self.eta1s):
            for j, eta2 in enumerate(self.eta2s):
                for k, eta3 in enumerate(self.eta3s):
                    policy = self.policy_dict[(eta1, eta2, eta3)]
                    policy_robot_weights, policy_human_weight = policy.get_predict_weights(time_t)
                    #print(time_t, policy_robot_weights, policy_human_weight)
                    robot_weights += policy_robot_weights * policy_weights[i,j, k]
                    human_weight += policy_human_weight * policy_weights[i,j, k]
        return robot_weights, human_weight
