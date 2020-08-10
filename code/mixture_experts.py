import numpy as np
import scipy.special
from statsmodels.tsa.arima.model import ARIMA

import torch
from torch import nn

from test_yelp import run_test


class Predictor:
    def __str__(self):
        return "predictor"


class OraclePredictor:
    def __init__(self, path_times, models, human_max_loss: float):
        self.weights = np.array([1])
        self.curr_num_experts = 1
        self.path_times = path_times
        self.models = models
        self.human_max_loss = human_max_loss

    def __str__(self):
        return "Oracle"

    def add_expert(self, time_t):
        if time_t == 0:
            return

        self.curr_num_experts += 1

    def update_weights(self, time_t, loss_t, prev_weights=None):
        path_time = self.path_times[time_t]
        curr_models = self.models[: time_t + 1]
        criterion = nn.L1Loss()
        oracle_loss_robot = np.array(
            [
                run_test(
                    model["model"],
                    path_time,
                    fields=model["fields"],
                    criterion=criterion,
                )
                for model in curr_models
            ]
        )

        if np.min(oracle_loss_robot) < self.human_max_loss:
            self.weights = np.array(
                oracle_loss_robot == np.min(oracle_loss_robot), dtype=float
            )
        else:
            weight = self.human_max_loss / np.min(oracle_loss_robot)
            self.weights = weight * np.array(
                oracle_loss_robot == np.min(oracle_loss_robot), dtype=float
            )

    def get_predict_weights(self, time_t):
        return self.weights, 1 - np.sum(self.weights)


class ExpWeightingWithHuman(Predictor):
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


class TimeTrendForecaster(Predictor):
    """
    Ordinary exponential weighting, modified to let in new experts
    First expert is human
    """

    def __init__(
        self,
        num_experts: int,
        eta: float,
        human_max_loss: float,
        order=(2, 1, 0),
        min_size: int = 7,
    ):
        self.human_max_loss = human_max_loss
        self.eta = eta
        self.num_experts = num_experts
        self.weights = np.ones(num_experts)
        self.curr_num_experts = 0
        self.order = order
        self.min_size = min_size

        self.loss_histories = np.zeros((num_experts, 1))

    def __str__(self):
        return "ARIMA_%d_%d_%d" % self.order

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
        print("loss histo", self.loss_histories)

    def get_predict_weights(self, time_t):
        predictions = np.zeros(self.curr_num_experts)
        for i in range(self.curr_num_experts):
            if i < time_t:
                if self.loss_histories[i].size > self.min_size:
                    try:
                        predictions[i] = self.fit_arima_get_output(
                            self.loss_histories[i]
                        )
                    except Exception as e:
                        print(e)
                        predictions[i] = np.mean(self.loss_histories[i])
                else:
                    # Use average until we can use ARIMA model?
                    predictions[i] = np.mean(self.loss_histories[i])
            else:
                predictions[i] = self.human_max_loss + 0.1

        projected_expert_losses = (
            np.sum(self.loss_histories[: self.curr_num_experts], axis=1) + predictions
        )
        projected_human_loss = self.human_max_loss * (time_t + 1)
        raw_weights = np.exp(
            -self.eta
            * np.concatenate([[projected_human_loss], projected_expert_losses])
        )
        all_weights = raw_weights / np.sum(raw_weights)
        return all_weights[1:], all_weights[0]

    def fit_arima_get_output(self, losses):
        arima_model = ARIMA(losses, order=self.order)
        res = arima_model.fit()
        print("FORE", res.forecast())
        return res.forecast(steps=1)[0]


class BlindWeight:
    def __init__(self):
        self.weights = np.array([1])
        self.curr_num_experts = 1

    def __str__(self):
        return "Blind"

    def add_expert(self, time_t):
        if time_t == 0:
            return

        # Set either to what our new_model_eta is or the same as the previously submitted model
        self.weights = np.array([0] * self.curr_num_experts + [1])
        self.curr_num_experts += 1

    def update_weights(self, time_t, loss_t, prev_weights=None):
        if time_t == 0:
            return self.weights
        return self.weights

    def get_predict_weights(self, time_t):
        return self.weights, 0


class ExpWeighting:
    """
    Ordinary exponential weighting, modified to let in new experts
    """

    def __init__(self, T, eta, new_model_eta):
        self.weights = np.array([1])
        self.T = T
        self.curr_num_experts = 1
        self.eta = eta
        self.new_model_eta = new_model_eta

    def __str__(self):
        return "Exp"

    def add_expert(self, time_t):
        if time_t == 0:
            return

        # Set either to what our new_model_eta is or the same as the previously submitted model
        new_model_eta = min(
            1 - self.new_model_eta, self.weights[-1] / (1 + self.weights[-1])
        )
        updated_weight = np.concatenate([self.weights, [0]])
        new_init_weights = np.array([0] * self.curr_num_experts + [1])
        self.weights = (
            1 - new_model_eta
        ) * updated_weight + new_model_eta * new_init_weights
        self.curr_num_experts += 1

    def update_weights(self, time_t, loss_t, prev_weights=None):
        if time_t == 0:
            return self.weights
        loss_t = np.sum(loss_t, axis=1)
        update_weight = np.exp(-self.eta * loss_t)
        numerator = update_weight * self.weights
        denom = np.sum(numerator)
        self.weights = numerator / denom

    def get_predict_weights(self, time_t):
        return self.weights


# class ExpWeightingWithHuman(ExpWeighting):
#    """
#    Ordinary exponential weighting, modified to let in new experts
#    First expert is human
#    """
#    def __init__(self, T, human_max_loss: float, eta_factor, new_model_eta, init_weights: np.ndarray = np.array([1,0])):
#        self.weights = init_weights
#        self.human_max_loss = human_max_loss
#        self.T = T
#        self.curr_num_experts = 2
#        self.eta_factor = eta_factor
#        self.eta = eta_factor/np.sqrt(T)
#        self.new_model_eta = new_model_eta
#
#    def __str__(self):
#        return "Exp_%.2f_%.2f" % (self.human_max_loss, self.eta_factor)
#
#    def add_expert(self, time_t):
#        if time_t == 0:
#            return
#
#        # Set either to what our new_model_eta is or the same as the previously submitted model
#        new_model_eta = 1 - self.new_model_eta #, self.weights[-1]/(1 + self.weights[-1]))
#        updated_weight = np.concatenate([self.weights, [0]])
#        new_init_weights = np.array([0] * self.curr_num_experts + [np.max(self.weights)])
#        self.weights = (1 - new_model_eta) * updated_weight + new_model_eta * new_init_weights
#        self.curr_num_experts += 1
#
#    def update_weights(self, time_t, indiv_robot_loss_t, prev_weights=None):
#        if time_t == 0:
#            return self.weights
#        batch_n = indiv_robot_loss_t.shape[1]
#        loss_t = np.concatenate([[batch_n * self.human_max_loss], np.sum(indiv_robot_loss_t, axis=1)])
#        #regret_t = np.sum(prev_weights * loss_t) - loss_t
#        update_weight = -self.eta * loss_t
#        print("UPDATE", update_weight)
#        self.weights = update_weight + self.weights
#
#    def get_predict_weights(self, time_t):
#        norm_weights = scipy.special.softmax(self.weights)
#        return norm_weights[1:], norm_weights[0]


class MetaExpWeighting:
    """
    Ordinary exponential weighting, modified to let in new experts
    """

    def __init__(self, T, eta, num_experts: int, forecaster_keys):
        self.curr_num_experts = num_experts
        self.forecaster_keys = forecaster_keys
        self.weights = np.ones(num_experts) / num_experts
        self.T = T
        self.eta = eta

    def __str__(self):
        return "Exp"

    def update_weights(self, time_t, loss_t, prev_weights=None):
        if time_t == 0:
            return self.weights
        loss_t = np.sum(loss_t, axis=1)
        update_weight = np.exp(-self.eta * loss_t)
        numerator = update_weight * self.weights
        denom = np.sum(numerator)
        self.weights = numerator / denom

    def get_predict_weights(self, time_t):
        return self.weights
