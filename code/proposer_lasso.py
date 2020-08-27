import numpy as np

from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV

from proposer import Proposer
from trial_data import TrialData


class LassoModel:
    def __init__(self, eps=1e-4, n_alphas=200, cv=5):
        self.model = LassoCV(eps=eps, n_alphas=n_alphas, cv=cv)
        #self.max_val = max_val
        #self.min_val = min_val

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, t=None):
        predictions = self.model.predict(X)
        return predictions
        #trunc_predictions = np.maximum(
        #    np.minimum(predictions, self.max_val), self.min_val
        #)
        #return trunc_predictions


    def loss(self, dataset):
        y_hat = self.predict(dataset.x).flatten()
        y = dataset.y.flatten()
        return np.power(y - y_hat, 2)

class LogisticRegressionCVWrap(LogisticRegressionCV):
    def predict(self, X, t=None):
        return super().predict(X)

    def loss(self, dataset):
        p_hat = self.predict_proba(dataset.x)
        #return yhat.flatten() != dataset.y.flatten()
        y = dataset.y.flatten()
        return -(np.log(p_hat[:,0]) * y + np.log(1 - p_hat[:,1]) * (1 - y))

class LassoProposer(Proposer):
    def __init__(
        self, sim_func_form: str, eps=1e-4, n_alphas=200, cv=5, max_val=1, min_val=0
    ):
        self.sim_func_form = sim_func_form
        self.eps = eps
        self.n_alphas = n_alphas
        self.cv = cv
        self.max_val = max_val
        self.min_val = min_val
        self.proposal_history = []

    def propose_model(self, trial_data: TrialData, curr_model_idx: int = None, do_append: bool = True):
        assert trial_data.num_batches == self.num_models + 1

        if self.sim_func_form == "gaussian":
            model = LassoModel(
                eps=self.eps,
                n_alphas=self.n_alphas,
                cv=self.cv,
                #max_val=self.max_val,
                #min_val=self.min_val,
            )
        elif self.sim_func_form == "bernoulli":
            model = LogisticRegressionCVWrap(
                cv=self.cv,
                # penalty='l1',
                # solver='liblinear',
                max_iter=1000,
            )
        cum_data = trial_data.get_start_to_end_data(0)
        print("NUM OBS", cum_data.num_obs)
        model.fit(cum_data.x, cum_data.y.flatten())

        if do_append:
            self.proposal_history.append(model)
        return model
