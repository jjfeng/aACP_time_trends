import numpy as np
import scipy.special

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

    def update_weights(self, time_t, loss_t, prev_weights = None):
        path_time = self.path_times[time_t]
        curr_models = self.models[:time_t + 1]
        criterion = nn.L1Loss()
        oracle_loss_robot = np.array([
                run_test(model['model'], path_time, fields=model['fields'], criterion=criterion) for model in curr_models])
        if np.min(oracle_loss_robot) < self.human_max_loss:
            self.weights = np.array(oracle_loss_robot == np.min(oracle_loss_robot), dtype=float)
        else:
            self.weights = np.zeros(oracle_loss_robot.shape)

    def get_predict_weights(self, time_t):
        return self.weights, 1 - np.sum(self.weights)

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

    def update_weights(self, time_t, loss_t, prev_weights = None):
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
        new_model_eta = min(1 - self.new_model_eta, self.weights[-1]/(1 + self.weights[-1]))
        updated_weight = np.concatenate([self.weights, [0]])
        new_init_weights = np.array([0] * self.curr_num_experts + [1])
        self.weights = (1 - new_model_eta) * updated_weight + new_model_eta * new_init_weights
        self.curr_num_experts += 1

    def update_weights(self, time_t, loss_t, prev_weights = None):
        if time_t == 0:
            return self.weights
        loss_t = np.sum(loss_t, axis=1)
        update_weight = np.exp(-self.eta * loss_t)
        numerator = update_weight * self.weights
        denom = np.sum(numerator)
        self.weights = numerator/denom

    def get_predict_weights(self, time_t):
        return self.weights


class ExpWeightingWithHuman(ExpWeighting):
    """
    Ordinary exponential weighting, modified to let in new experts
    First expert is human
    """
    def __init__(self, T, human_max_loss: float, eta_factor, new_model_eta, init_weights: np.ndarray = np.array([1,0])):
        self.weights = init_weights
        self.human_max_loss = human_max_loss
        self.T = T
        self.curr_num_experts = 2
        self.eta_factor = eta_factor
        self.eta = eta_factor/np.sqrt(T)
        self.new_model_eta = new_model_eta

    def __str__(self):
        return "Exp_%.2f_%.2f" % (self.human_max_loss, self.eta_factor)

    def add_expert(self, time_t):
        if time_t == 0:
            return

        # Set either to what our new_model_eta is or the same as the previously submitted model
        new_model_eta = 1 - self.new_model_eta #, self.weights[-1]/(1 + self.weights[-1]))
        updated_weight = np.concatenate([self.weights, [0]])
        new_init_weights = np.array([0] * self.curr_num_experts + [np.max(self.weights)])
        self.weights = (1 - new_model_eta) * updated_weight + new_model_eta * new_init_weights
        self.curr_num_experts += 1

    def update_weights(self, time_t, indiv_robot_loss_t, prev_weights=None):
        if time_t == 0:
            return self.weights
        batch_n = indiv_robot_loss_t.shape[1]
        loss_t = np.concatenate([[batch_n * self.human_max_loss], np.sum(indiv_robot_loss_t, axis=1)])
        #regret_t = np.sum(prev_weights * loss_t) - loss_t
        update_weight = -self.eta * loss_t
        print("UPDATE", update_weight)
        self.weights = update_weight + self.weights

    def get_predict_weights(self, time_t):
        norm_weights = scipy.special.softmax(self.weights)
        return norm_weights[1:], norm_weights[0]

#class ExpWeightWithHumanPredictorConstraints(ExpWeightingWithPredictor):
#    """
#    Add constraint checking
#    """
#    def __init__(self, predictor: Predictor, T, batch_n: int, human_max_loss: float, eta, new_model_eta, prediction_upweight :float  = 1, alpha:float=0.05):
#        self.predictor = predictor
#        self.human_max_loss = human_max_loss
#        self.weights = np.array([0.1, 0.9])
#        self.batch_n = batch_n
#        self.T = T
#        self.curr_num_experts = 2
#        self.eta = eta
#        self.new_model_eta = new_model_eta
#        self.prediction_upweight = prediction_upweight
#        self.alpha = alpha
#        self.tot_losses = 0
#        self.actual_losses = 0
#
#    def __str__(self):
#        return "Exp%s" % str(self.predictor)
#
#    def add_expert(self, time_t):
#        self.predictor.add_expert()
#        if time_t == 0:
#            return self.weights
#
#        # Set either to what our new_model_eta is or the same as the previously submitted model
#        new_model_eta = (1 - self.new_model_eta)
#        updated_weight = np.concatenate([self.weights, [0]])
#        new_init_weights = np.array([0] * self.curr_num_experts + [1])
#        self.weights = (1 - new_model_eta) * updated_weight + new_model_eta * new_init_weights
#
#        self.curr_num_experts += 1
#
#    def update_weights(self, time_t, indiv_loss_robot_t = None, prev_weights = None):
#        if time_t == 0:
#            return self.weights
#        batch_n = indiv_loss_robot_t.shape[1]
#        loss_t = np.concatenate([
#            [batch_n * self.human_max_loss],
#            np.sum(indiv_loss_robot_t, axis=1)])
#        if loss_t is not None:
#            self.tot_losses += np.sum(prev_weights * loss_t)
#            self.actual_losses += np.sum(prev_weights[1:] * loss_t[1:])
#        self.predictor.update_estimates(indiv_loss_robot_t)
#
#        update_weight = np.exp(-self.eta * loss_t)
#        numerator = update_weight * self.weights
#        denom = np.sum(numerator)
#        self.weights = numerator/denom
#
#    def get_predict_weights(self, time_t):
#        _pred_loss, _upper_bound = self.predictor.predict_loss(time_t)
#        pred_loss = self.batch_n * _pred_loss
#        upper_bound = self.batch_n * _upper_bound
#        mean_loss_prediction = (self.actual_losses + upper_bound)/(time_t + 1)/self.batch_n
#        good_mask = mean_loss_prediction < self.alpha
#        if np.sum(good_mask) == 0:
#            return np.zeros(self.weights.size - 1), 1
#        # if np.sum(good_mask) != good_mask.size:
#            # print("   SUBSET:", np.sum(good_mask), good_mask.size)
#
#        update_weight = np.exp(-self.eta * self.prediction_upweight * np.concatenate([[self.batch_n * self.human_max_loss], pred_loss]))
#        numerator = update_weight * self.weights
#        ml_numerator = numerator[1:]
#        ml_numerator[np.logical_not(good_mask)] = 0
#        denom = numerator[0] + np.sum(ml_numerator)
#        return ml_numerator/denom, numerator[0]/denom
