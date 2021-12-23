import os
import pickle
import unittest
from pathlib import Path

import numpy as np
import ray.rllib.agents.marwil as marwil
from ray.rllib.agents.ppo import PPOTrainer
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


class TestMARWILRNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=4)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_marwil_rnn(self):

        ModelCatalog.register_custom_model("rnn", RNNSpyModel)
        rllib_dir = Path(__file__).parent.parent.parent.parent
        print("rllib dir={}".format(rllib_dir))
        data_file = os.path.join(rllib_dir, "tests/data/cartpole/large.json")
        print("data_file={} exists={}".format(data_file, os.path.isfile(data_file)))

        config = marwil.DEFAULT_CONFIG.copy()
        config["num_workers"] = 2
        config["evaluation_num_workers"] = 1
        config["evaluation_interval"] = 3
        config["evaluation_duration"] = 5
        config["evaluation_parallel_to_training"] = True
        # Evaluate on actual environment.
        config["evaluation_config"] = {"input": "sampler"}
        config["model"] = {
            "custom_model": "rnn",
            "max_seq_len": 4,
            "vf_share_layers": True,
        }
        # Learn from offline data.
        config["input"] = [data_file]
        num_iterations = 10
        min_reward = 70.0

        # Test for all frameworks.
        frameworks = "tf"
        for _ in framework_iterator(config, frameworks=frameworks):
            trainer = marwil.MARWILTrainer(config=config, env="CartPole-v0")
            learnt = False
            for i in range(num_iterations):
                results = trainer.train()
                check_train_results(results)
                print(results)

                eval_results = results.get("evaluation")
                if eval_results:
                    print(
                        "iter={} R={} ".format(i, eval_results["episode_reward_mean"])
                    )
                    # Learn until some reward is reached on an actual live env.
                    if eval_results["episode_reward_mean"] > min_reward:
                        print("learnt!")
                        learnt = True
                        break

            if not learnt:
                raise ValueError(
                    "MARWILTrainer did not reach {} reward from expert "
                    "offline data!".format(min_reward)
                )

            check_compute_single_action(trainer, include_prev_action_reward=True)

            trainer.stop()


def test_marwil_rnn(self=None):

    ModelCatalog.register_custom_model("rnn", RNNModel)
    rllib_dir = Path(__file__).parent.parent.parent.parent
    print("rllib dir={}".format(rllib_dir))
    data_file = os.path.join(rllib_dir, "tests/data/cartpole/large.json")
    print("data_file={} exists={}".format(data_file, os.path.isfile(data_file)))

    config = marwil.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1
    config["evaluation_num_workers"] = 1
    config["evaluation_interval"] = 3
    config["evaluation_duration"] = 5
    config["evaluation_parallel_to_training"] = True
    # Evaluate on actual environment.
    config["evaluation_config"] = {"input": "sampler"}
    config["model"] = {
        "custom_model": "rnn",
        "max_seq_len": 4,
        "vf_share_layers": True,
    }
    config["input_evaluation"] = []  # ["is", "wis"]
    # Learn from offline data.
    config["input"] = [data_file]
    num_iterations = 10
    min_reward = 70.0

    # Test for all frameworks.
    frameworks = "tf"
    for _ in framework_iterator(config, frameworks=frameworks):
        trainer = marwil.MARWILTrainer(config=config, env="CartPole-v0")
        learnt = False
        for i in range(num_iterations):
            results = trainer.train()
            check_train_results(results)
            print(results)

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
                "MARWILTrainer did not reach {} reward from expert "
                "offline data!".format(min_reward)
            )

        check_compute_single_action(trainer, include_prev_action_reward=True)

        trainer.stop()


if __name__ == "__main__":
    # import sys

    # import pytest

    # sys.exit(pytest.main(["-v", __file__]))
    test_marwil_rnn()
