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

class OptimisticFixedShare(OptimisticPolicy):
    """
    optimistic (jenky jean version?) Fixed Share
    First expert is human
    """

    def __init__(
            self, num_experts: int, eta: float, human_max_loss: float, time_trend_predictor, alpha: float = 0.1
    ):
        assert eta > 0
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.alpha = alpha
        self.num_experts = num_experts
        self.curr_num_experts = 0
        # initialized with only human weight
        self.weights = np.ones(1)
        self.v_weights = np.ones(1)
        self.time_trend_predictor = time_trend_predictor

        self.loss_histories = np.zeros((num_experts, 1))
        print("alpha", self.alpha)

    def __str__(self):
        return "OptimisticFixedShare"

    def add_expert(self, time_t):
        self.curr_num_experts += 1
        self.weights = np.concatenate([self.weights, [0]])
        self.v_weights = np.concatenate([self.v_weights, [0]])

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
        self.v_weights = self.weights * np.exp(- self.eta * np.concatenate([[self.human_max_loss], model_losses_t]))
        self.weights = (1 - self.alpha) * self.v_weights + self.alpha * np.mean(self.v_weights)
        # Adding normalization to prevent numerical underflow
        self.weights /= np.max(self.weights)
        print("self wei", self.weights)

    def get_predict_weights(self, time_t: int):
        predicted_expert_losses = np.array(
            [
                self.fit_and_predict(self.loss_histories[i])
                for i in range(self.curr_num_experts)
            ]
        )

        num_steps = self.loss_histories.shape[1]
        raw_multipliers = np.exp(
            -self.eta
            * np.concatenate([[self.human_max_loss], predicted_expert_losses])
        )
        raw_weights = self.v_weights * raw_multipliers
        raw_weights[-1] = 0
        all_weights = raw_weights / np.sum(raw_weights)
        print("orig weights", self.weights/np.sum(self.weights))
        print("optim weights", all_weights)
        return all_weights[1:], all_weights[0]

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
        self.weights /= np.max(self.weights)
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
        human_loss = self.human_max_loss * time_t
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
