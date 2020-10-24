import time
import logging
import numpy as np
import scipy.stats
from numpy import ndarray
from typing import List
import pandas as pd

from common import score_mixture_model
from approval_history import ApprovalHistory


class Simulation:
    """"""

    def __init__(
        self, nature, proposer, policy, human_max_loss, num_test_obs: int = 1000
    ):
        self.nature = nature
        self.proposer = proposer
        self.policy = policy
        self.human_max_loss = human_max_loss
        self.num_test_obs = num_test_obs
        self.total_time = nature.total_time - 1
        self.batch_model_preds = []
        self.batch_targets = []
        self.criterion = self.proposer.criterion
        self.approval_hist = ApprovalHistory(
            human_max_loss=self.human_max_loss, policy_name=str(self.policy)
        )


    def run(self, hook_func):
        for t in range(self.total_time):
            logging.info("TIME STEP %d", t)
            print("TIME", t)

            # Let the policy adapt based on data
            self.policy.add_expert(t)
            self.policy.update_weights(
                t,
                self.criterion,
                self.batch_model_preds[-1] if t > 0 else None,
                self.batch_targets[-1] if t > 0 else None,
            )
            robot_weights, human_weight = self.policy.get_predict_weights(t)

            # Monitoring data
            sub_trial_data = self.nature.get_trial_data(t + 1)
            obs_batch_data = sub_trial_data.batch_data[-1]
            batch_preds, batch_target = self.proposer.get_model_preds_and_target(
                obs_batch_data
            )
            self.batch_model_preds.append(batch_preds)
            self.batch_targets.append(batch_target)

            # Population data
            pop_batch_data = self.nature.create_test_data(t + 1, self.num_test_obs)
            if pop_batch_data is not None:
                (
                    pop_batch_preds,
                    pop_batch_target,
                ) = self.proposer.get_model_preds_and_target(pop_batch_data)
            else:
                pop_batch_preds = batch_preds
                pop_batch_target = batch_target

            # Score models
            policy_loss_t = score_mixture_model(
                human_weight,
                robot_weights,
                self.criterion,
                batch_preds,
                batch_target,
                self.human_max_loss,
            )
            pop_policy_loss_t = score_mixture_model(
                human_weight,
                robot_weights,
                self.criterion,
                pop_batch_preds,
                pop_batch_target,
                self.human_max_loss,
            )

            self.approval_hist.append(
                human_weight,
                robot_weights,
                policy_loss_t,
                pop_policy_loss_t,
            )

            # Let nature adapt if it wants
            self.nature.next(self.approval_hist)

            logging.info("losses %s", pop_policy_loss_t)
            logging.info(
                "robot weights %s (max %d)", robot_weights, np.argmax(robot_weights)
            )
            logging.info("human weight %f", human_weight)

            if t < self.total_time - 1:
                self.proposer.propose_model(sub_trial_data, self.approval_hist)

            hook_func(self.approval_hist)
