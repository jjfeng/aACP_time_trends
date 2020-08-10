import numpy as np


class Policy:
    def __init__(self, human_max_loss: float):
        self.human_max_loss = human_max_loss

    def add_expert(self, time_t):
        self.curr_num_experts += 1

    def get_predict_weights(self, time_t: int):
        raise NotImplementedError()


class OptimisticMirrorDescent(Policy):
    """
    Ordinary exponential weighting, modified to let in new experts
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

        num_steps = self.loss_histories.shape[1] + 1
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

    def fit_and_predict(self, losses):
        return self.time_trend_predictor.forecast(losses)
