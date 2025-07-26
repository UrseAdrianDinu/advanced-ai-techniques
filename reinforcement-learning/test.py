import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from PIL import Image
from src.crafter_wrapper import Env
import random
import itertools
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
from dqn import DQNAgent
from ddqn import DDQNAgent
from categorical_dqn import CategoricalDQNAgent
from rainbow_dqn import RainbowDQNAgent
from simple_rainbow_dqn import SimpleRainbowDQNAgent

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
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs, crt_step)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward
    

    _save_stats(episodic_returns, crt_step, opt.logdir, writer)


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


def test_model(agent, env, num_episodes=10):
    """Test the trained model for a specified number of episodes using a greedy policy.
    
    Args:
        agent (DQNAgent): The trained agent.
        env (Env): The environment to test in.
        num_episodes (int): Number of episodes to run the test.
    
    Returns:
        list: A list of accumulated rewards for each episode.
    """
    agent.q_network.eval()  # Set the network to evaluation mode
    accumulated_rewards = []

    with torch.no_grad():
        for episode in range(num_episodes):
            obs, done = env.reset(), False
            episode_reward = 0
            while not done:
                action = agent.act_greedy(obs)  # Use the greedy action for testing
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            accumulated_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Accumulated Reward = {episode_reward}")
    
    avg_reward = np.mean(accumulated_rewards)
    min_reward = np.min(accumulated_rewards)
    max_reward = np.max(accumulated_rewards)
    std_reward = np.std(accumulated_rewards)

    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")
    print(f"Min Reward: {min_reward}, Max Reward: {max_reward}, Std Dev: {std_reward}")

    return {
        "average_reward": avg_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "std_reward": std_reward
    }    


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env("eval", opt)
    input_shape = (opt.history_length, 84, 84)
    num_actions = env.action_space.n
    writer = SummaryWriter(log_dir=opt.logdir)

    # Initialize agent and load checkpoint
    agent = SimpleRainbowDQNAgent(input_shape, num_actions, writer)
    checkpoint_path = r"C:\Users\bebed\OneDrive\Desktop\AAIT\a\crafter_starting_code\simple_rainbow_checkpoint_1000000.pth"  # Path to your checkpoint
    checkpoint = torch.load(checkpoint_path)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.target_network.load_state_dict(checkpoint['target_network'])
    print(f"Loaded model from {checkpoint_path}")
    agent.q_network.eval()
    agent.target_network.eval()

    # Evaluate the loaded agent
    num_test_episodes = 100
    res = test_model(agent, env, num_episodes=num_test_episodes)


    


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
