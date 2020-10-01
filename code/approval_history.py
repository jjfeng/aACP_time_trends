import numpy as np


class ApprovalHistory:
    def __init__(self, human_max_loss: float, policy_name: str):
        self.human_max_loss = human_max_loss
        self.policy_name = policy_name
        self.policy_loss_history = []
        self.all_loss_history = []
        self.human_history = []
        self.expert_weights_history = []

    @property
    def size(self):
        return len(self.human_history)

    def append(
        self,
        human_weight: float,
        expert_weights: np.ndarray,
        loss: float,
        all_loss: np.ndarray,
    ):
        self.human_history.append(human_weight)
        self.expert_weights_history.append(expert_weights)
        self.policy_loss_history.append(loss)
        self.all_loss_history.append(all_loss)

    def __str__(self):
        return """
        Policy loss: %s (total: %.2f)
        Human hist: %s (mean: %.2f)
        """ % (
            np.array_str(np.array(self.policy_loss_history), precision=2),
            np.sum(self.policy_loss_history),
            np.array_str(np.array(self.human_history), precision=2),
            np.mean(self.human_history),
        )
