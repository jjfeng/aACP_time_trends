"""
These classes just prefetch model predictions and targets
"""
import numpy as np


class ModelPredsAndTargets:
    def __init__(self):
        self.model_preds = []
        self.targets = []

    def append(self, new_model_preds, new_targets):
        self.model_preds.append(new_model_preds)
        self.targets.append(new_targets)


class AggModelPredsAndTargets:
    def __init__(self):
        self.model_preds = []
        self.targets = None

    @property
    def tot_time(self):
        return len(self.targets) if self.targets is not None else None

    @property
    def num_models(self):
        return self.model_preds[-1].shape[0] if len(self.model_preds) else 0

    def append(self, model_preds_and_targets: ModelPredsAndTargets):
        if self.targets is not None:
            for curr_target, target in zip(
                self.targets, model_preds_and_targets.targets
            ):
                assert np.all(np.isclose(target, curr_target))
        else:
            self.targets = model_preds_and_targets.targets

        if self.num_models == 0:
            self.model_preds = model_preds_and_targets.model_preds
        else:
            for time_t in range(self.num_models, self.tot_time):
                print("old", self.model_preds[time_t].shape)
                self.model_preds[time_t] = np.concatenate(
                    [
                        self.model_preds[time_t],
                        model_preds_and_targets.model_preds[time_t],
                    ]
                )
                print("new", self.model_preds[time_t].shape)
