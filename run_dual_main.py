import os
print(os.getcwd())
import sys
project_path = './'
sys.path.append("/usr/local/webots/lib")
sys.path.insert(0, project_path + 'pytorch')
print(sys.path)

from envs.abb_assembly_env.Env_robot_assembly import env_assembly_search
# from envs.webots_assembly_env.simulation_env import ArmEnv
import argparse
import numpy as np
from code.pytorch.utils.solver import Assembly_solver


def test_env(env):
    env.reset()
    state = np.random.rand(22)
    print(env.set_robot(state) - state)
    while True:
        env.render()


def main(env, args):
    solver = Assembly_solver(args, env, project_path)
    if not args.eval_only:
        solver.train()
    else:
        solver.eval_only()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default='TD3')  # Policy name
    parser.add_argument("--env_name", default="dual-peg-in-hole")  # OpenAI gym environment name
    parser.add_argument("--log_path", default='transfer/dual_assembly_VPB_new')

    parser.add_argument("--eval_only", default=True)
    parser.add_argument("--render", default=False)
    parser.add_argument("--save_video", default=False)
    parser.add_argument("--video_size", default=(600, 400))

    parser.add_argument("--load_model", default=True)
    parser.add_argument("--load_path", default='./results/transfer/dual_assembly_VPB_new/TD3_dual-peg-in-hole_seed_0')
    parser.add_argument("--save_all_policy", default=True)
    parser.add_argument("--load_policy_idx", default=23400, type=int)
    parser.add_argument("--evaluate_Q_value", default=False)
    parser.add_argument("--reward_name", default='r_s')

    parser.add_argument("--seq_len", default=2, type=int)
    parser.add_argument("--ini_seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=100, type=int)  # How many time steps purely random policy is run for

    parser.add_argument("--eval_freq", default=600, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=4500, type=int)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--state_noise", default=0, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.2, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_episode_steps", default=150, type=int)

    parser.add_argument("--average_steps", default=20, type=int)
    parser.add_argument("--eval_episodes", default=30, type=int)

    args = parser.parse_args()

    env = env_assembly_search(step_max=args.max_episode_steps,
                              fuzzy=True,
                              add_noise=False)

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
        for i in range(0, 1):
            args.seed = i
            main(env, args)
