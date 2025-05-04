import argparse
from pathlib import Path
import datetime
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack

from env_wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import MarioAgent
from logger import MetricLogger

def make_env(skip, shape, stack):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, gym_super_mario_bros.actions.COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    env = FrameStack(env, num_stack=stack)
    return env

def main():
    env = make_env(4, 84, 4)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    max_step = 5000

    episodes = 400000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        for step in range(max_step):

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done: 
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

if __name__ == '__main__':
    main()