from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.use_active_masks = "active_masks" in self.batch.scheme
        self.use_masks = "masks" in self.batch.scheme
        self.use_bad_masks = "bad_masks" in self.batch.scheme
        self.agent_mask_template = np.ones((self.args.n_agents, 1), dtype=np.float32)

        pre_transition_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self._append_initial_mask_fields(pre_transition_data)
        self.batch.update(pre_transition_data, ts=0)

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated and not self._is_bad_transition(env_info),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            next_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self._append_step_mask_fields(next_transition_data, terminated, env_info)
            self.batch.update(next_transition_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
                
        if test_mode:
            if len(self.test_returns) == self.args.test_nepisode:
                self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_median", np.median(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def _append_initial_mask_fields(self, data):
        avail_actions = data["avail_actions"][0]
        if self.use_active_masks:
            data["active_masks"] = [self._infer_active_masks(avail_actions)]
        if self.use_masks:
            data["masks"] = [self.agent_mask_template.copy()]
        if self.use_bad_masks:
            data["bad_masks"] = [self.agent_mask_template.copy()]

    def _append_step_mask_fields(self, data, terminated, env_info):
        avail_actions = data["avail_actions"][0]
        if self.use_active_masks:
            data["active_masks"] = [self._infer_active_masks(avail_actions)]
        if self.use_masks:
            if terminated:
                data["masks"] = [np.zeros_like(self.agent_mask_template)]
            else:
                data["masks"] = [self.agent_mask_template.copy()]
        if self.use_bad_masks:
            if terminated and self._is_bad_transition(env_info):
                data["bad_masks"] = [np.zeros_like(self.agent_mask_template)]
            else:
                data["bad_masks"] = [self.agent_mask_template.copy()]

    def _infer_active_masks(self, avail_actions):
        avail_actions = np.asarray(avail_actions)
        return (avail_actions.sum(axis=-1, keepdims=True) > 1).astype(np.float32)

    def _is_bad_transition(self, env_info):
        return bool(env_info.get("bad_transition", False) or env_info.get("episode_limit", False))
