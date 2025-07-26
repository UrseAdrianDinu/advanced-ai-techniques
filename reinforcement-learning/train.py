import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
from PIL import Image
from src.crafter_wrapper import Env
import random
import itertools
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math

from dqn import DQNAgent
from ddqn import DDQNAgent
from categorical_dqn import CategoricalDQNAgent
from rainbow_dqn import RainbowDQNAgent
from simple_rainbow_dqn import SimpleRainbowDQNAgent
from policy_based import ReinforceAgent

def _save_stats(episodic_returns, crt_step, path, writer):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    writer.add_scalar('Eval/Avg Reward', avg_return, crt_step)

    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt, writer):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    agent.q_network.eval()
    episodic_returns = []
    with torch.no_grad():
        for _ in range(opt.eval_episodes):
            obs, done = env.reset(), False
            episodic_returns.append(0)
            while not done:
                action = agent.act_greedy(obs)
                obs, reward, done, info = env.step(action)
                episodic_returns[-1] += reward
    
    agent.q_network.train()
    _save_stats(episodic_returns, crt_step, opt.logdir, writer)


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {opt.device}")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    opt.target_update_interval = 10_000
    opt.batch_size = 64
    num_actions = env.action_space.n
    input_shape = (opt.history_length, 84, 84)

    writer = SummaryWriter(log_dir=opt.logdir)
    agent = CategoricalDQNAgent(input_shape, num_actions, writer)
    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    episode_rewards = 0

    while step_cnt < opt.steps or not done:
        if done:
            writer.add_scalar('Episode Reward', episode_rewards, ep_cnt)
            
            ep_cnt += 1
            episode_rewards = 0
            obs, done = env.reset(), False


        action = agent.act(obs, step_cnt)
        next_obs, reward, done, info = env.step(action)
        episode_rewards += reward  # Accumulate rewards for the current episode
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.update(opt.batch_size, step_cnt)
        obs = next_obs
        
        step_cnt += 1

        if step_cnt % 200_000 == 0:
            torch.save({
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict()
            }, f"{opt.logdir}/checkpoint_{step_cnt}.pth")

        if step_cnt % opt.target_update_interval == 0:
            agent.update_target_network()

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt, writer)

        if step_cnt % 100_000 == 0:
            torch.cuda.empty_cache()

    
def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
