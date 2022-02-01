from collections import defaultdict
import glob
import json
import logging
from gym import Space
import numpy as np
import os
from pathlib import Path
import random
import re
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import zipfile


from ray.rllib.utils.typing import EnvObsType




try:
    from smart_open import smart_open
except ImportError:
    smart_open = None

from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch, \
    SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.evaluation.collectors.simple_list_collector import _AgentCollector
from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.spaces.space_utils import clip_action, get_dummy_batch_for_space, normalize_action
from ray.rllib.evaluation.collectors.simple_list_collector import \
    SimpleListCollector
from ray.rllib.utils.typing import  AgentID, EpisodeID, EnvID, PolicyID, TensorType, FileType, SampleBatchType
from ray.rllib.evaluation.sample_batch_builder import \
    MultiAgentSampleBatchBuilder
from ray.rllib.env.base_env import  convert_to_base_env

logger = logging.getLogger(__name__)

WINDOWS_DRIVES = [chr(i) for i in range(ord("c"), ord("z") + 1)]


@PublicAPI
class JsonReader(InputReader):
    """Reader object that loads experiences from JSON file chunks.

    The input files will be read from in random order.
    """

    @PublicAPI
    def __init__(self,
                 inputs: Union[str, List[str]],
                 ioctx: Optional[IOContext] = None):
        """Initializes a JsonReader instance.

        Args:
            inputs: Either a glob expression for files, e.g. `/tmp/**/*.json`,
                or a list of single file paths or URIs, e.g.,
                ["s3://bucket/file.json", "s3://bucket/file2.json"].
            ioctx: Current IO context object or None.
        """

        self.ioctx = ioctx or IOContext()
        self.default_policy = self.policy_map = None
        if self.ioctx.worker is not None:
            from ray.rllib.evaluation.sampler import NewEpisodeDefaultDict

            self.policy_map = self.ioctx.worker.policy_map
            self.default_policy = self.policy_map.get(DEFAULT_POLICY_ID)
            # Whenever we observe a new episode+agent, add a new
            # _SingleTrajectoryCollector.

            config =  self.ioctx.config
            self.sample_collector = SimpleListCollector(
                policy_map=self.policy_map,
                clip_rewards= config["clip_rewards"],
                callbacks =config["callbacks"] ,
                multiple_episodes_in_batch=config["batch_mode"]=="truncate_episodes",
                rollout_fragment_length=config["rollout_fragment_length"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            )


            self._active_episodes : Dict[EnvID, Episode] = NewEpisodeDefaultDict(self.new_episode)

        if isinstance(inputs, str):
            inputs = os.path.abspath(os.path.expanduser(inputs))
            if os.path.isdir(inputs):
                inputs = [
                    os.path.join(inputs, "*.json"),
                    os.path.join(inputs, "*.zip")
                ]
                logger.warning(
                    f"Treating input directory as glob patterns: {inputs}")
            else:
                inputs = [inputs]

            if any(
                    urlparse(i).scheme not in [""] + WINDOWS_DRIVES
                    for i in inputs):
                raise ValueError(
                    "Don't know how to glob over `{}`, ".format(inputs) +
                    "please specify a list of files to read instead.")
            else:
                self.files = []
                for i in inputs:
                    self.files.extend(glob.glob(i))
        elif isinstance(inputs, (list, tuple)):
            self.files = list(inputs)
        else:
            raise ValueError(
                "type of inputs must be list or str, not {}".format(inputs))
        if self.files:
            logger.info("Found {} input files.".format(len(self.files)))
        else:
            raise ValueError("No files found matching {}".format(inputs))
        self.cur_file = None
        self._cur_eps_id = None

    def add_init_obs(self, sub_batch: Dict, episode_id:int, agent_key, env_id=0) -> None:
        #TODO (fcouthouis): handle multipolicy
        policy = self.default_policy
        agent_index = sub_batch[SampleBatch.AGENT_INDEX]
        agent_key = agent_index
        t = sub_batch[SampleBatch.T]
        init_obs = sub_batch[SampleBatch.OBS]

        # Add initial obs to Trajectory.
        assert agent_key not in self.agent_collectors
        # TODO: determine exact shift-before based on the view-req shifts.
        self.agent_collectors[agent_key] = _AgentCollector(
            policy.view_requirements, policy)

        
        self.agent_collectors[agent_key].add_init_obs(
            episode_id=episode_id,
            agent_index=agent_index,
            env_id=env_id,
            t=t,
            init_obs=init_obs)

    def new_episode(self, episode_id):
        worker = self.ioctx.worker
        # Pool of batch builders, which can be shared across episodes to pack
        # trajectory data.
        batch_builder_pool: List[MultiAgentSampleBatchBuilder] = []

        def get_batch_builder():
            if batch_builder_pool:
                return batch_builder_pool.pop()
            else:
                return None
        extra_batch_callback = lambda x: None
        env_id=0
        episode = Episode(
            worker.policy_map,
            worker.policy_mapping_fn,
            get_batch_builder,
            extra_batch_callback,
            env_id=env_id,
            worker=worker,
            episode_id=episode_id,
        )
        return episode
    def collect_batch(self, batch: SampleBatch):
        print('batch obs',batch[SampleBatch.OBS])
        for sub_batch in batch.rows():
            #TODO: change this?
            agent_id = sub_batch[SampleBatch.AGENT_INDEX]
            eps_id = sub_batch[SampleBatch.EPS_ID]

            is_new_episode: bool = eps_id not in self._active_episodes
            episode: Episode = self._active_episodes[eps_id]

            if not is_new_episode:
                episode._add_agent_rewards({agent_id:sub_batch[SampleBatch.REWARDS]})
            
            agent_done = sub_batch[SampleBatch.DONES]
            last_observation: EnvObsType = episode.last_observation_for(
                agent_id)
            policy_id: PolicyID = episode.policy_for(agent_id)

            # A new agent (initial obs) is already done -> Skip entirely.
            if last_observation is None and agent_done:
                continue

            agent_id  = sub_batch[SampleBatch.AGENT_INDEX]
            #TODO: is this new_obs?
            filtered_obs = sub_batch[SampleBatch.OBS]
            raw_obs = sub_batch[SampleBatch.OBS]
            episode._set_last_observation(agent_id, filtered_obs)
            episode._set_last_raw_obs(agent_id, raw_obs)
            episode._set_last_done(agent_id, agent_done)
            episode._set_last_info(agent_id, sub_batch[SampleBatch.INFOS])

            
            # Record transition info if applicable.
            if last_observation is None:
                self.sample_collector.add_init_obs(episode, agent_id, episode.env_id,
                                              policy_id, episode.length - 1,
                                              filtered_obs)
            else:
                del sub_batch[SampleBatch.OBS]
                self.sample_collector.add_action_reward_next_obs(
                    episode.episode_id, agent_id, episode.env_id, policy_id,
                    agent_done, sub_batch)

            # if all_agents_done:
            #     ma_sample_batch = sample_collector.postprocess_episode(
            #     episode,
            #     is_done=is_done or (hit_horizon and not soft_horizon),
            #     check_dones=check_dones,
            #     build=not multiple_episodes_in_batch)
            input_dict = self.sample_collector.get_inference_input_dict(policy_id)
            print('INFERENCE INPUT DICT',input_dict)
            print('state_in_0: ', input_dict['state_in_0'])

            # fill sub_batch with missing rnn states
            states = defaultdict(list)
            for key, value in input_dict.items():
                if key not in sub_batch:
                    states[key].append(value)
        
        for state_key, state_value in states.items():
            batch[state_key] = np.concatenate(state_value, axis=-1)

    def add_rnn_states(self, sub_batch: Dict):
        policy = self.default_policy
        keys = [0]

        buffers = {}
        for k in keys:
            collector = self.agent_collectors[k]
            buffers[k] = collector.buffers
        # Use one agent's buffer_structs (they should all be the same).
        buffer_structs = self.agent_collectors[keys[0]].buffer_structs

        input_dict = {}
        for view_col, view_req in policy.view_requirements.items():
            # Not used for action computations.
            if not view_req.used_for_compute_actions:
                continue

            # Create the batch of data from the different buffers.
            data_col = view_req.data_col or view_col
            delta = -1 if data_col in [
                SampleBatch.OBS, SampleBatch.ENV_ID, SampleBatch.EPS_ID,
                SampleBatch.AGENT_INDEX, SampleBatch.T
            ] else 0
            # Range of shifts, e.g. "-100:0". Note: This includes index 0!
            if view_req.shift_from is not None:
                time_indices = (view_req.shift_from + delta,
                                view_req.shift_to + delta)
            # Single shift (e.g. -1) or list of shifts, e.g. [-4, -1, 0].
            else:
                time_indices = view_req.shift + delta

            # Loop through agents and add up their data (batch).
            data = None
            for k in keys:
                # Buffer for the data does not exist yet: Create dummy
                # (zero) data.
                if data_col not in buffers[k]:
                    if view_req.data_col is not None:
                        space = policy.view_requirements[
                            view_req.data_col].space
                    else:
                        space = view_req.space

                    if isinstance(space, Space):
                        fill_value = get_dummy_batch_for_space(
                            space,
                            batch_size=0,
                        )
                    else:
                        fill_value = space

                    self.agent_collectors[k]._build_buffers({
                        data_col: fill_value
                    })
                    print('NOT IN data_col,',data_col)

                print('data_col,',data_col, 'data:',data)
                if data is None:
                    data = [[] for _ in range(len(buffers[keys[0]][data_col]))]

                # `shift_from` and `shift_to` are defined: User wants a
                # view with some time-range.
                if isinstance(time_indices, tuple):
                    # `shift_to` == -1: Until the end (including(!) the
                    # last item).
                    if time_indices[1] == -1:
                        for d, b in zip(data, buffers[k][data_col]):
                            d.append(b[time_indices[0]:])
                    # `shift_to` != -1: "Normal" range.
                    else:
                        for d, b in zip(data, buffers[k][data_col]):
                            d.append(b[time_indices[0]:time_indices[1] + 1])
                # Single index.
                else:
                    for d, b in zip(data, buffers[k][data_col]):
                        d.append(b[time_indices])
            print(data_col, "data",type(data),data)
            np_data = [np.array(d) for d in data]
            if data_col in buffer_structs:
                input_dict[view_col] = tree.unflatten_as(
                    buffer_structs[data_col], np_data)
            else:
                input_dict[view_col] = np_data[0]
        print("**input_dict",input_dict)
        for key,value in input_dict.items():
            if key not in sub_batch:
                sub_batch[key] = value
        print("**sub_batch",sub_batch)

    @override(InputReader)
    def next(self) -> SampleBatchType:
        batch = self._try_parse(self._next_line())    

        tries = 0
        while not batch and tries < 100:
            tries += 1
            logger.debug("Skipping empty line in {}".format(self.cur_file))
            batch = self._try_parse(self._next_line())
        if not batch:
            raise ValueError(
                "Failed to read valid experience batch from file: {}".format(
                    self.cur_file))

        return self._postprocess_if_needed(batch)

    def read_all_files(self) -> SampleBatchType:
        """Reads through all files and yields one SampleBatchType per line.

        When reaching the end of the last file, will start from the beginning
        again.

        Yields:
            One SampleBatch or MultiAgentBatch per line in all input files.
        """
        for path in self.files:
            file = self._try_open_file(path)
            while True:
                line = file.readline()
                if not line:
                    break
                batch = self._try_parse(line)
                if batch is None:
                    break
                yield batch

    def _postprocess_if_needed(self,
                               batch: SampleBatchType) -> SampleBatchType:
        
        if not self.ioctx.config.get("postprocess_inputs"):
            return batch

        # RNN case without states in json file
        if "state_in_0" in self.default_policy.view_requirements:
            # No state in data
            if "state_in_0" not in batch.keys():
                self.collect_batch(batch)
                print('new batch after',batch)

        if isinstance(batch, SampleBatch):
            out = []
            for sub_batch in batch.split_by_episode():
                out.append(
                    self.default_policy.postprocess_trajectory(sub_batch))
            return SampleBatch.concat_samples(out)
        else:
            # TODO(ekl) this is trickier since the alignments between agent
            #  trajectories in the episode are not available any more.
            raise NotImplementedError(
                "Postprocessing of multi-agent data not implemented yet.")

    def _try_open_file(self, path):
        if urlparse(path).scheme not in [""] + WINDOWS_DRIVES:
            if smart_open is None:
                raise ValueError(
                    "You must install the `smart_open` module to read "
                    "from URIs like {}".format(path))
            ctx = smart_open
        else:
            # Allow shortcut for home directory ("~/" -> env[HOME]).
            if path.startswith("~/"):
                path = os.path.join(os.environ.get("HOME", ""), path[2:])

            # If path doesn't exist, try to interpret is as relative to the
            # rllib directory (located ../../ from this very module).
            path_orig = path
            if not os.path.exists(path):
                path = os.path.join(Path(__file__).parent.parent, path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Offline file {path_orig} not found!")

            # Unzip files, if necessary and re-point to extracted json file.
            if re.search("\\.zip$", path):
                with zipfile.ZipFile(path, "r") as zip_ref:
                    zip_ref.extractall(Path(path).parent)
                path = re.sub("\\.zip$", ".json", path)
                assert os.path.exists(path)
            ctx = open
        file = ctx(path, "r")
        return file

    def _try_parse(self, line: str) -> Optional[SampleBatchType]:
        line = line.strip()
        if not line:
            return None
        try:
            batch = self._from_json(line)
        except Exception:
            logger.exception("Ignoring corrupt json record in {}: {}".format(
                self.cur_file, line))
            return None

        # Clip actions (from any values into env's bounds), if necessary.
        cfg = self.ioctx.config
        if cfg.get("clip_actions") and self.ioctx.worker is not None:
            if isinstance(batch, SampleBatch):
                batch[SampleBatch.ACTIONS] = clip_action(
                    batch[SampleBatch.ACTIONS],
                    self.default_policy.action_space_struct)
            else:
                for pid, b in batch.policy_batches.items():
                    b[SampleBatch.ACTIONS] = clip_action(
                        b[SampleBatch.ACTIONS],
                        self.ioctx.worker.policy_map[pid].action_space_struct)
        # Re-normalize actions (from env's bounds to zero-centered), if
        # necessary.
        if cfg.get("actions_in_input_normalized") is False and \
                self.ioctx.worker is not None:

            # If we have a complex action space and actions were flattened
            # and we have to normalize -> Error.
            error_msg = \
                "Normalization of offline actions that are flattened is not "\
                "supported! Make sure that you record actions into offline " \
                "file with the `_disable_action_flattening=True` flag OR " \
                "as already normalized (between -1.0 and 1.0) values. " \
                "Also, when reading already normalized action values from " \
                "offline files, make sure to set " \
                "`actions_in_input_normalized=True` so that RLlib will not " \
                "perform normalization on top."

            if isinstance(batch, SampleBatch):
                pol = self.default_policy
                if isinstance(pol.action_space_struct, (tuple, dict)) and \
                        not pol.config.get("_disable_action_flattening"):
                    raise ValueError(error_msg)
                batch[SampleBatch.ACTIONS] = normalize_action(
                    batch[SampleBatch.ACTIONS], pol.action_space_struct)
            else:
                for pid, b in batch.policy_batches.items():
                    pol = self.policy_map[pid]
                    if isinstance(pol.action_space_struct, (tuple, dict)) and \
                            not pol.config.get("_disable_action_flattening"):
                        raise ValueError(error_msg)
                    b[SampleBatch.ACTIONS] = normalize_action(
                        b[SampleBatch.ACTIONS],
                        self.ioctx.worker.policy_map[pid].action_space_struct)
        return batch

    def _next_line(self) -> str:
        if not self.cur_file:
            self.cur_file = self._next_file()
        line = self.cur_file.readline()
        tries = 0
        while not line and tries < 100:
            tries += 1
            if hasattr(self.cur_file, "close"):  # legacy smart_open impls
                self.cur_file.close()
            self.cur_file = self._next_file()
            line = self.cur_file.readline()
            if not line:
                logger.debug("Ignoring empty file {}".format(self.cur_file))
        if not line:
            raise ValueError("Failed to read next line from files: {}".format(
                self.files))
        return line

    def _next_file(self) -> FileType:
        # If this is the first time, we open a file, make sure all workers
        # start with a different one if possible.
        if self.cur_file is None and self.ioctx.worker is not None:
            idx = self.ioctx.worker.worker_index
            total = self.ioctx.worker.num_workers or 1
            path = self.files[round((len(self.files) - 1) * (idx / total))]
        # After the first file, pick all others randomly.
        else:
            path = random.choice(self.files)
        return self._try_open_file(path)

    def _from_json(self, data: str) -> SampleBatchType:
        if isinstance(data, bytes):  # smart_open S3 doesn't respect "r"
            data = data.decode("utf-8")
        json_data = json.loads(data)

        # Try to infer the SampleBatchType (SampleBatch or MultiAgentBatch).
        if "type" in json_data:
            data_type = json_data.pop("type")
        else:
            raise ValueError("JSON record missing 'type' field")

        if data_type == "SampleBatch":
            if self.ioctx.worker is not None and \
                    len(self.ioctx.worker.policy_map) != 1:
                raise ValueError(
                    "Found single-agent SampleBatch in input file, but our "
                    "PolicyMap contains more than 1 policy!")
            for k, v in json_data.items():
                json_data[k] = unpack_if_needed(v)
            if self.ioctx.worker is not None:
                policy = next(iter(self.ioctx.worker.policy_map.values()))
                json_data = self._adjust_obs_actions_for_policy(
                    json_data, policy)
            batch = SampleBatch(json_data)
            print('SampleBatchJson',batch)
        elif data_type == "MultiAgentBatch":
            policy_batches = {}
            for policy_id, policy_batch in json_data["policy_batches"].items():
                inner = {}
                for k, v in policy_batch.items():
                    inner[k] = unpack_if_needed(v)
                if self.ioctx.worker is not None:
                    policy = self.ioctx.worker.policy_map[policy_id]
                    inner = self._adjust_obs_actions_for_policy(inner, policy)
                policy_batches[policy_id] = SampleBatch(inner)
            batch = MultiAgentBatch(policy_batches, json_data["count"])
        else:
            raise ValueError(
                "Type field must be one of ['SampleBatch', 'MultiAgentBatch']",
                data_type)

        # Adjust the seq-lens array depending on the incoming agent sequences.
        if self.default_policy.is_recurrent():
            # Add seq_lens & max_seq_len to batch
            seq_lens = []
            max_seq_len = self.default_policy.config["model"]["max_seq_len"]
            count = batch.count
            while count > 0:
                seq_lens.append(min(count, max_seq_len))
                count -= max_seq_len
            batch[SampleBatch.SEQ_LENS] = np.array(seq_lens)
            batch.max_seq_len = max_seq_len
        print('json batch keys ',batch.keys())
        return batch
    def _adjust_obs_actions_for_policy(self, json_data: dict,
                                       policy: Policy) -> dict:
        """Handle nested action/observation spaces for policies.

        Translates nested lists/dicts from the json into proper
        np.ndarrays, according to the (nested) observation- and action-
        spaces of the given policy.

        Providing nested lists w/o this preprocessing step would
        confuse a SampleBatch constructor.
        """
        for k, v in policy.view_requirements.items():
            if k not in json_data:
                continue
            if policy.config.get("_disable_action_flattening") and \
                    (k == SampleBatch.ACTIONS or
                     v.data_col == SampleBatch.ACTIONS):
                json_data[k] = tree.map_structure_up_to(
                    policy.action_space_struct,
                    lambda comp: np.array(comp),
                    json_data[k],
                    check_types=False,
                )
            elif policy.config.get("_disable_preprocessor_api") and \
                    (k == SampleBatch.OBS or
                     v.data_col == SampleBatch.OBS):
                json_data[k] = tree.map_structure_up_to(
                    policy.observation_space_struct,
                    lambda comp: np.array(comp),
                    json_data[k],
                    check_types=False,
                )
        return json_data
