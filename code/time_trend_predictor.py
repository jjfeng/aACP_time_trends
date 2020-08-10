import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class TimeTrendPredictor:
    def fit(self, data: np.ndarray):
        raise NotImplementedError

    def forecast(self):
        return 0


class ARIMAPredictor(TimeTrendPredictor):
    def __init__(self, order: tuple, min_size: int, max_loss: float):
        self.order = order
        self.min_size = 7
        self.max_loss = max_loss

    def forecast(self, data: np.ndarray):
        if data.size > 1:
            if losses.size > self.min_size:
                try:
                    arima_model = ARIMA(losses, order=self.order)
                    res = arima_model.fit()
                    res = res.forecast(steps=1)[0]
                except Exception as e:
                    res = np.mean(losses)
            else:
                # Use average until we can use ARIMA model?
                res = np.mean(losses)
        else:
            res = self.max_loss
        return res
