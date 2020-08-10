import numpy as np


class ApprovalHistory:
    def __init__(self):
        self.policy_loss_history = []
        self.all_loss_history = []
        self.human_history = []

    def append(self, human_weight: float, loss: float, all_loss: np.ndarray):
        self.human_history.append(human_weight)
        self.policy_loss_history.append(loss)
        self.all_loss_history.append(all_loss)

    def __str__(self):
        return """
        Policy loss: %s
        Human hist: %s
        """ % (
            np.array_str(np.array(self.policy_loss_history), precision=2),
            np.array_str(np.array(self.human_history), precision=2),
        )
