import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import copy as cp
import time

import code.tensorflow_code.common.tf_util as U
from code.tensorflow_code.common.tf_util import load_variables, save_variables
from code.tensorflow_code.common import logger
from code.tensorflow_code.common.schedules import LinearSchedule

from code.tensorflow_code import deepq
from code.tensorflow_code.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from code.tensorflow_code.deepq.utils import ObservationInput

from code.tensorflow_code.common.tf_util import get_session
from code.tensorflow_code.deepq.models import build_q_func
from envs.abb_assembly_env.Env_robot_assembly import env_assembly_search


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          total_episodes=100,
          total_steps=50,
          buffer_size=5000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=10,
          checkpoint_freq=100,
          checkpoint_path=None,
          learning_starts=1000,
          learning_times=10,
          gamma=0.99,
          target_network_update_freq=10,
          prioritized_replay=True,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          nb_epochs=5,
          nb_epoch_cycles=30,
          nb_rollout_steps=150,
          nb_train_steps=60,
          param_noise=False,
          callback=None,
          restore=False,
          load_path=None,
          data_path=None,
          **network_kwargs
          ):
    # Create all the functions necessary to train the model

    sess = get_session()

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_dim,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_dim,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = []
    epoch_episode_states = []
    epoch_episode_times = []
    epoch_episode_actions = []
    saved_mean_reward = None
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        t = 0
        for i in range(nb_epoch_cycles):
            obs, _, _ = env.reset()
            episode_reward = 0.
            start_time = time.time()
            episode_states = []
            reset = True
            for j in range(nb_rollout_steps):

                if callback is not None:
                    if callback(locals(), globals()):
                        break

                kwargs = {}
                if not param_noise:
                    update_eps = exploration.value(t)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(
                        1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True

                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]

                env_action = action

                reset = False

                # move to next step
                new_obs, original_state, reward, done, safe_or_not, executeAction = env.step_discrete_action(action)

                if safe_or_not is False:
                    break

                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, reward, new_obs, float(done))
                episode_states.append(cp.deepcopy(original_state))
                epoch_episode_actions.append(cp.deepcopy(executeAction))
                obs = new_obs
                episode_reward += reward

                if done:
                    # obs = env.reset()
                    # reset = True
                    # logger.info("================== The Search Phase has Finished!!! ===================")
                    break

                if t > learning_starts and t % train_freq == 0:

                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    for learning_term in range(learning_times):
                        if prioritized_replay:
                            experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                            weights, batch_idxes = np.ones_like(rewards), None
                        td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                        if prioritized_replay:
                            new_priorities = np.abs(td_errors) + prioritized_replay_eps
                            replay_buffer.update_priorities(batch_idxes, new_priorities)

                if t > learning_starts and t % target_network_update_freq == 0:
                    # Update target network periodically.
                    update_target()

                mean_100ep_reward = round(np.mean(episode_rewards[-10:-1]), 1)
                num_episodes = len(episode_rewards)
                if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

                if (checkpoint_freq is not None and t > learning_starts and
                        num_episodes > 100 and t % checkpoint_freq == 0):
                    if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                        if print_freq is not None:
                            logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                saved_mean_reward, mean_100ep_reward))
                        save_variables(model_file)
                        model_saved = True
                        saved_mean_reward = mean_100ep_reward
                t += 1

            episode_time = time.time() - start_time
            epoch_episode_states.append(cp.deepcopy(episode_states))
            episode_rewards.append(cp.deepcopy(episode_reward))
            epoch_episode_times.append(cp.deepcopy(episode_time))

            np.save('./data_fuzzy_test_new/episode_rewards', episode_rewards)
            np.save('./data_fuzzy_test_new/episode_state', epoch_episode_states)
            np.save('./data_fuzzy_test_new/episode_time', epoch_episode_times)
            np.save('./data_fuzzy_test_new/episode_actions', epoch_episode_actions)

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)


if __name__ == '__main__':

    env = env_assembly_search(step_max=150,
                              fuzzy=True,
                              add_noise=True)
    data_path = './ddpg_data/'
    model_path = './ddpg_model/'
    nb_epochs = 5
    nb_epoch_cycles = 30
    nb_rollout_steps = 200
    file_name = '_epochs_' + str(nb_epochs) + "_episodes_" + str(nb_epoch_cycles) + "_rollout_steps_" + str(nb_rollout_steps)

    learn(network='mlp',
          env=env,
          restore=False,
          nb_epochs=nb_epochs,
          nb_epoch_cycles=nb_epoch_cycles,
          nb_rollout_steps=nb_rollout_steps,
          nb_train_steps=60,
)