import scipy.stats
import numpy as np

from trial_data import TrialData
from proposer import Proposer
from approval_history import ApprovalHistory


class FakeBernoulliModel:
    def __init__(self, data_gen, eps, decay=0, offset=0):
        self.data_gen = data_gen
        self.eps = eps
        self.decay = decay
        self.offset = offset
        # assert eps <= 1 and eps >= 0

    def predict(self, x, t):
        true_mus = self.data_gen.mu_func(x)
        eps0, eps1 = self.get_eps(t)
        new_mus0 = ((1 - eps0) * true_mus + eps0 * 0.5) * (true_mus < 0.5)
        new_mus1 = ((1 - eps1) * true_mus + eps1 * 0.5) * (true_mus >= 0.5)
        new_mus = new_mus0 + new_mus1
        return new_mus
        #return scipy.stats.bernoulli.rvs(p=new_mus)

    def get_eps(self, t):
        eps0 = min(1, max(0, self.eps[0] - self.decay * np.sin(t * self.offset)))
        eps1 = min(1, max(0, self.eps[1] - self.decay * np.sin(t * self.offset)))
        return eps0, eps1

    def loss(self, dataset):
        p_hat = self.predict(dataset.x, t=0).flatten()
        #return yhat.flatten() != dataset.y.flatten()
        y = dataset.y.flatten()
        return -(np.log(p_hat) * y + np.log(1 - p_hat) * (1 - y))


class RandomProposer(Proposer):
    """
    This produces the same exact models over time with injected noise
    """

    def __init__(self, data_gen, noise=0.2, increment=0, decay=0, offset_scale=0):
        """
        @param increment: how much worse to decrease the true positive, true negative performance by
        """
        assert data_gen.sim_func_form == "bernoulli"

        self.proposal_history = []
        self.approval_history = []
        self.data_gen = data_gen
        self.noise = noise
        self.increment = increment
        self.decay = decay
        self.offset_scale = offset_scale

    def propose_model(self, trial_data, approval_hist: ApprovalHistory = None):
        if approval_hist.size == 0:
            noise = np.ones(2) * self.noise
        else:
            noise = np.ones(2) * max(self.noise + np.sin(-len(self.proposal_history) * self.increment) * self.increment, 0)
        model = FakeBernoulliModel(
            self.data_gen,
            noise,
            self.decay,
            offset=np.random.rand() * self.offset_scale,
        )
        self.proposal_history.append(model)
        return model
