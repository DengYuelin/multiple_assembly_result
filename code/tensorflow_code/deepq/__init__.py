from code.tensorflow_code.deepq import models  # noqa
from code.tensorflow_code.deepq.build_graph import build_act, build_train  # noqa
from code.tensorflow_code.deepq.deepq import learn, load_act  # noqa
from code.tensorflow_code.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from code.tensorflow_code.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=True)
