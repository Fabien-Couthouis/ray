"""
TensorFlow policy class used for CQL.
"""
import logging
from calendar import c
from functools import partial
from typing import Dict, List, Tuple, Type, Union

import gym
import numpy as np
import ray.experimental.tf_utils
import tree
# from ray.rllib.agents.sac.sac_tf_policy import (
#     apply_gradients as sac_apply_gradients,
#     compute_and_clip_gradients as sac_compute_and_clip_gradients,
#     get_distribution_inputs_and_class as sac_get_distribution_inputs_and_class,
#     _get_dist_class,
#     build_sac_model,
#     postprocess_trajectory,
#     setup_late_mixins,
#     stats,
#     validate_spaces,
#     ActorCriticOptimizerMixin as SACActorCriticOptimizerMixin,
#     ComputeTDErrorMixin,
#     TargetNetworkMixin,
# )
from ray.rllib.agents.dqn.dqn_tf_policy import (PRIO_WEIGHTS,
                                                ComputeTDErrorMixin, QLoss,
                                                TargetNetworkMixin)
from ray.rllib.agents.dqn.dqn_tf_policy import \
    adam_optimizer as dqn_adam_optimizer
from ray.rllib.agents.dqn.dqn_tf_policy import (build_q_losses, build_q_model,
                                                build_q_stats)
from ray.rllib.agents.dqn.dqn_tf_policy import \
    clip_gradients as dqn_clip_gradients
from ray.rllib.agents.dqn.dqn_tf_policy import compute_q_values
from ray.rllib.agents.dqn.dqn_tf_policy import \
    get_distribution_inputs_and_class as dqn_get_distribution_inputs_and_class
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.agents.dqn.dqn_tf_policy import \
    setup_late_mixins as dqn_setup_late_mixins
from ray.rllib.agents.dqn.dqn_tf_policy import \
    setup_mid_mixins as dqn_setup_mid_mixins
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (Categorical,
                                                TFActionDistribution)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import (get_variable, try_import_tf,
                                       try_import_tfp)
from ray.rllib.utils.typing import (LocalOptimizer, ModelGradients, TensorType,
                                    TrainerConfigDict)

import ray

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)

MEAN_MIN = -9.0
MEAN_MAX = 9.0

def _repeat_tensor(t: TensorType, n: int):
    # Insert new axis at position 1 into tensor t
    t_rep = tf.expand_dims(t, 1)
    # Repeat tensor t_rep along new axis n times
    multiples = tf.concat([[1, n], tf.tile([1], tf.expand_dims(tf.rank(t) - 1, 0))], 0)
    t_rep = tf.tile(t_rep, multiples)
    # Merge new axis into batch axis
    t_rep = tf.reshape(t_rep, tf.concat([[-1], tf.shape(t)[1:]], 0))
    return t_rep


# Returns policy tiled actions and log probabilities for CQL Loss
def policy_actions_repeat(model, action_dist, obs, num_repeat=1):
    batch_size = tf.shape(tree.flatten(obs)[0])[0]
    obs_temp = tree.map_structure(lambda t: _repeat_tensor(t, num_repeat), obs)
    logits = model.get_policy_output(obs_temp)
    policy_dist = action_dist(logits, model)
    actions, logp_ = policy_dist.sample_logp()
    logp = tf.expand_dims(logp_, -1)
    return actions, tf.reshape(logp, [batch_size, num_repeat, 1])


def q_values_repeat(model, obs, actions, twin=False):
    action_shape = tf.shape(actions)[0]
    obs_shape = tf.shape(tree.flatten(obs)[0])[0]
    num_repeat = action_shape // obs_shape
    obs_temp = tree.map_structure(lambda t: _repeat_tensor(t, num_repeat), obs)
    if not twin:
        preds_ = model.get_q_values(obs_temp, actions)
    else:
        preds_ = model.get_twin_q_values(obs_temp, actions)
    preds = tf.reshape(preds_, [obs_shape, num_repeat, 1])
    return preds

def cql_dqn_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TFActionDistribution],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    logger.info(f"Current iteration = {policy.cur_iter}")
    policy.cur_iter += 1

    """Constructs the loss for DQNTFPolicy.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        train_batch (SampleBatch): The training data.

    Returns:
        TensorType: A single loss tensor.
    """
    config = policy.config

    # CQL Parameters
    cql_temp = config["temperature"]
    min_q_weight = config["min_q_weight"]

    # Q Loss
    # q network evaluation
    q_t, q_logits_t, q_dist_t, _ = compute_q_values(
        policy,
        model,
        SampleBatch({"obs": train_batch[SampleBatch.CUR_OBS]}),
        state_batches=None,
        explore=False,
    )

    # target q network evalution
    q_tp1, _q_logits_tp1, q_dist_tp1, _ = compute_q_values(
        policy,
        policy.target_model,
        SampleBatch({"obs": train_batch[SampleBatch.NEXT_OBS]}),
        state_batches=None,
        explore=False,
    )
    if not hasattr(policy, "target_q_func_vars"):
        policy.target_q_func_vars = policy.target_model.variables()

    # q scores for actions which we know were selected in the given state.
    one_hot_selection = tf.one_hot(
        tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32), policy.action_space.n
    )
    q_t_selected = tf.reduce_sum(q_t * one_hot_selection, 1)
    q_logits_t_selected = tf.reduce_sum(
        q_logits_t * tf.expand_dims(one_hot_selection, -1), 1
    )

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        (
            q_tp1_using_online_net,
            q_logits_tp1_using_online_net,
            q_dist_tp1_using_online_net,
            _,
        ) = compute_q_values(
            policy,
            model,
            SampleBatch({"obs": train_batch[SampleBatch.NEXT_OBS]}),
            state_batches=None,
            explore=False,
        )
        q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = tf.one_hot(
            q_tp1_best_using_online_net, policy.action_space.n
        )
        q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = tf.reduce_sum(
            q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1
        )
    else:
        q_tp1_best_one_hot_selection = tf.one_hot(
            tf.argmax(q_tp1, 1), policy.action_space.n
        )
        q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
        q_dist_tp1_best = tf.reduce_sum(
            q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1
        )

    policy.q_loss = QLoss(
        q_t_selected,
        q_logits_t_selected,
        q_tp1_best,
        q_dist_tp1_best,
        train_batch[PRIO_WEIGHTS],
        train_batch[SampleBatch.REWARDS],
        tf.cast(train_batch[SampleBatch.DONES], tf.float32),
        config["gamma"],
        config["n_step"],
        config["num_atoms"],
        config["v_min"],
        config["v_max"],
    )
    q_loss = policy.q_loss.loss

    # CQL Loss (We are using Entropy version of CQL (the best version))
    cql_loss = (
        tf.reduce_mean(tf.reduce_logsumexp(q_t / cql_temp, axis=1))
        * min_q_weight
        * cql_temp
    )
    cql_loss = cql_loss - (tf.reduce_mean(q_t) * min_q_weight)
    

    policy.cql_loss = [cql_loss]
    total_loss = q_loss + cql_loss 
    policy.total_loss = [total_loss]
    return total_loss

    


def cql_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    stats_dict = build_q_stats(policy, train_batch)
    stats_dict["cql_loss"] = tf.reduce_mean(tf.stack(policy.cql_loss))
    stats_dict["total_loss"] = tf.reduce_mean(tf.stack(policy.total_loss))
    return stats_dict


def setup_early_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> None:
    """Call mixin classes' constructors before Policy's initialization.

    Adds the necessary optimizers to the given Policy.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    policy.cur_iter = 0
    # if condition:
    # ActorCriticOptimizerMixin.__init__(policy, config)
    dqn_setup_mid_mixins(policy,obs_space=obs_space, action_space=action_space,config=config)



def validate_spaces(
    policy: Policy,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> None:
    """Validates the observation- and action spaces used for the Policy.

    Args:
        policy (Policy): The policy, whose spaces are being validated.
        observation_space (gym.spaces.Space): The observation space to
            validate.
        action_space (gym.spaces.Space): The action space to validate.
        config (TrainerConfigDict): The Policy's config dict.

    Raises:
        UnsupportedSpaceException: If one of the spaces is not supported.
    """
    # if not isinstance(action_space, Discrete):
    #     return sac_validate_spaces(policy=policy, observation_space=observation_space,action_space=action_space, config=config)
    return None

def setup_late_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> None:
    dqn_setup_late_mixins(policy=policy,obs_space=obs_space,action_space=action_space,config=config)


def build_model(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> ModelV2:
    """Constructs the necessary ModelV2 for the Policy and returns it.

    Args:
        policy (Policy): The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config (TrainerConfigDict): The SAC trainer's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    return build_q_model(
    policy,
    obs_space,
    action_space,
    config,
)

def build_mixins():
    # if condition:
    #     return [ActorCriticOptimizerMixin, TargetNetworkMixin, ComputeTDErrorMixin]
    return [
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ]
    

def get_distribution_inputs_and_class(
    policy: Policy,
    model: ModelV2,
    input_dict: SampleBatch,
    *,
    explore: bool = True,
    **kwargs
) -> Tuple[TensorType, Type[TFActionDistribution], List[TensorType]]:
    """The action distribution function to be used the algorithm.

    An action distribution function is used to customize the choice of action
    distribution class and the resulting action distribution inputs (to
    parameterize the distribution object).
    After parameterizing the distribution, a `sample()` call
    will be made on it to generate actions.

    Args:
        policy (Policy): The Policy being queried for actions and calling this
            function.
        model (SACTFModel): The SAC specific Model to use to generate the
            distribution inputs (see sac_tf|torch_model.py). Must support the
            `get_policy_output` method.
        obs_batch (TensorType): The observations to be used as inputs to the
            model.
        explore (bool): Whether to activate exploration or not.

    Returns:
        Tuple[TensorType, Type[TFActionDistribution], List[TensorType]]: The
            dist inputs, dist class, and a list of internal state outputs
            (in the RNN case).
    """
    # if condition:
    #    obs_batch = input_dict[SampleBatch.CUR_OBS]
    #     return sac_get_distribution_inputs_and_class()
    return dqn_get_distribution_inputs_and_class(policy,model,input_dict,explore=explore)

# Build a child class of `TFPolicy`, given the custom functions defined
# above.
CQLDQNTFPolicy = build_tf_policy(
    name="CQLDQNTFPolicy",
    loss_fn=cql_dqn_loss,
    get_default_config=lambda: ray.rllib.agents.cql.cql_dqn.CQL_DQN_DEFAULT_CONFIG,
    validate_spaces=validate_spaces,
    stats_fn=cql_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    after_init=setup_late_mixins,
    make_model=build_model,
    mixins=build_mixins(),
    action_distribution_fn=get_distribution_inputs_and_class,
    compute_gradients_fn=dqn_clip_gradients,
    # DQN relative
    before_loss_init=setup_early_mixins, #instead of before_init
    optimizer_fn=dqn_adam_optimizer,
    extra_action_out_fn=lambda policy: {"q_values": policy.q_values},
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.q_loss.td_error},
)
