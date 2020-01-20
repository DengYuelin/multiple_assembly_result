import os
print(os.getcwd())
import sys
project_path = './'
sys.path.append("/usr/local/webots/lib")
sys.path.insert(0, project_path + 'pytorch')
print(sys.path)
from envs import ArmEnv
import argparse
import numpy as np
from code.pytorch.utils.solver import Solver


def test_env(env):
    env.reset()
    state = np.random.rand(22)
    print(env.set_robot(state) - state)
    while True:
        env.render()


def main(env, args):
    solver = Solver(args, env, project_path)
    if not args.eval_only:
        solver.train()
    else:
        solver.eval_only()


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()

    # path, env and policy
    parser.add_argument("--policy_name", default='ATD3_RNN')  # Policy name
    parser.add_argument("--env_name", default="HopperBulletEnv-v0")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='runs/single_peg_in_hole_assembly')

    # basic settings
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e3, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)  # Max time steps to run environment for
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate

    # para for entropy
    parser.add_argument("--entropy_alpha", default=0.2, type=float)  # for gaussian policy
    parser.add_argument("--entropy_alpha_h", default=0.005, type=float)  # for option policy

    # para for HRL
    parser.add_argument("--weighted_action", default=True)
    parser.add_argument("--option_num", default=8, type=int)

    parser.add_argument("--option_buffer_size", default=5000, type=int)  # Batch size for both actor and critic
    parser.add_argument("--option_batch_size", default=50, type=int)  # Batch size for both actor and critic
    parser.add_argument("--policy_batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--critic_batch_size", default=400, type=int)  # Batch size for both actor and critic

    # save and load policy
    parser.add_argument("--load_policy", default=False)
    parser.add_argument("--load_policy_idx", default=1000000, type=int)
    parser.add_argument("--save_all_policy", default=True)
    parser.add_argument("--save_policy_inx", default=1000000, type=int)

    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--render", default=False)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--video_size", default=(600, 400))

    parser.add_argument("--evaluate_Q_value", default=False)
    parser.add_argument("--reward_name", default='r_s')
    parser.add_argument("--seq_len", default=2, type=int)

    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=0, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates

    args = parser.parse_args()

    env = ArmEnv()
    policy_name_vec = ['TD3']

    # average_steps = [5, 10, 20, 40]
    # for policy_name in policy_name_vec:
    #     for num_steps in average_steps:
    #         args.average_steps = num_steps
    #         for i in range(0, 5):
    #             args.policy_name = policy_name
    #             args.seed = i
    #             main(env, args)

    for policy_name in policy_name_vec:
        args.policy_name = policy_name
        for i in range(0, 5):
            args.seed = i
            main(env, args)
