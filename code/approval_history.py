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
