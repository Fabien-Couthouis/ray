import os
import pickle
import unittest
from pathlib import Path

import numpy as np
import ray.rllib.agents.ppo as ppo
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.examples.env.debug_counter_env import DebugCounterEnv
from ray.rllib.examples.models.rnn_spy_model import RNNSpyModel
from ray.rllib.examples.models.rnn_model import RNNModel
from ray.rllib.models import ModelCatalog
from ray.rllib.offline import JsonReader
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import (
    check,
    check_compute_single_action,
    check_train_results,
    framework_iterator,
)
from ray.tune.registry import register_env

import ray

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()



def test_ppo_rnn(self=None):
    ModelCatalog.register_custom_model("rnn", RNNModel)
    # The path may change depending on the location of this file (works for rllib.agents.ppo.tests)
    rllib_dir = Path(__file__).parent.parent.parent.parent
    print("rllib dir={}".format(rllib_dir))
    data_file = os.path.join(rllib_dir, "tests/data/cartpole/large.json")
    print("data_file={} exists={}".format(data_file, os.path.isfile(data_file)))

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 0
    config["evaluation_num_workers"] = 0
    # Evaluate on actual environment.
    config["evaluation_config"] = {"input": "sampler"}

    config["model"] = {
        "custom_model": "rnn",
    }
    # config["input_evaluation"] = [] #["is", "wis"]
    # Learn from offline data.
    # config["input"] = [data_file]
    num_iterations = 10
    min_reward = 70.0

    frameworks = "tf"
    for _ in framework_iterator(config, frameworks=frameworks):
        trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
        learnt = False
        for i in range(num_iterations):
            results = trainer.train()
            check_train_results(results)

            eval_results = results.get("evaluation")
            if eval_results:
                print("iter={} R={} ".format(i, eval_results["episode_reward_mean"]))
                # Learn until some reward is reached on an actual live env.
                if eval_results["episode_reward_mean"] > min_reward:
                    print("learnt!")
                    learnt = True
                    break

        if not learnt:
            raise ValueError(
                "PPOTrainer did not reach {} reward from expert "
                "offline data!".format(min_reward)
            )

        check_compute_single_action(trainer, include_prev_action_reward=True)

        trainer.stop()


if __name__ == "__main__":
    test_ppo_rnn()
