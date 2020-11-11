import numpy as np


class ApprovalHistory:
    def __init__(self, human_max_loss: float, policy_name: str):
        self.human_max_loss = human_max_loss
        self.policy_name = policy_name
        self.policy_loss_history = []
        self.expected_policy_loss_history = []
        self.human_history = []
        self.expert_weights_history = []

    @property
    def size(self):
        return len(self.human_history)

    def append(
        self,
        human_weight: float,
        expert_weights: np.ndarray,
        obs_loss: float,
        expected_loss: float,
    ):
        self.human_history.append(human_weight)
        self.expert_weights_history.append(expert_weights)
        self.policy_loss_history.append(obs_loss)
        self.expected_policy_loss_history.append(expected_loss)

    def __str__(self):
        return """
        %s
        Policy loss: %s (total: %.2f)
        Human hist: %s (mean: %.2f)
        """ % (
            self.policy_name,
            np.array_str(np.array(self.expected_policy_loss_history), precision=2),
            np.sum(self.expected_policy_loss_history),
            np.array_str(np.array(self.human_history), precision=2),
            np.mean(self.human_history),
        )

    @property
    def currently_approved_idx(self):
        """
        @return average idx of the currently approved robot (normalized by robot weight,
        not total weight)
        """
        robot_weights = self.expert_weights_history[-1]
        if np.sum(robot_weights) > 0:
            return np.sum(np.arange(self.size) * robot_weights)/robot_weights.sum()
        else:
            return None

    @property
    def last_approved_idx(self):
        """
        @return average idx of the previously approved robot (normalized by robot weight,
        not total weight)
        """
        for i in range(2, len(self.expert_weights_history) + 1):
            robot_weights = self.expert_weights_history[-i]
            if np.sum(robot_weights) > 0:
                return np.sum(np.arange(robot_weights.size) * robot_weights)/robot_weights.sum()
        return None

