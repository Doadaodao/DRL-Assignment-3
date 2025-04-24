import gym
import torch
import numpy as np
from collections import deque
from pathlib import Path
from torchvision import transforms as T

# make sure your PYTHONPATH lets you import MarioNet from your train.py
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from train import MarioNet

def permute_orientation(observation):
    # permute [H, W, C] array to [C, H, W] tensor
    observation = np.transpose(observation, (2, 0, 1))
    observation = torch.tensor(observation.copy(), dtype=torch.float)
    return observation

def greyscale(observation):
    observation = permute_orientation(observation)
    transform = T.Grayscale()
    observation = transform(observation)
    return observation

def resize(observation, shape):
    transforms = T.Compose(
        [T.Resize(shape, antialias=True), T.Normalize(0, 255)]
    )
    observation = transforms(observation).squeeze(0)
    return observation

class Agent(object):
    """Loads a trained MarioNet checkpoint and applies the same
    SkipFrame/GrayScale/Resize/FrameStack pipeline in-act."""
    def __init__(self):
        # action space must match JoypadSpace(COMPLEX_MOVEMENT)
        self.action_space = gym.spaces.Discrete(12)

        # device & model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MarioNet(input_dim=(4, 84, 84), output_dim=12).to(self.device)

        # ── load your checkpoint (edit this path!) ───────────────────────
        ckpt_path = "./mario_net_23.chkpt"
        ckpt_path = "./checkpoints/2025-04-22T14-16-27/mario_net_23.chkpt"
        ckpt_path = "./checkpoints/2025-04-22T15-01-10/mario_net_33.chkpt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        # ─────────────────────────────────────────────────────────────────

        self.step = 0
        self.skip = 4
        self.last_action = 0
        self.done = False

        # a deque to hold our 4-frame stack
        self.frame_stack = deque(maxlen=4)
        

    def act(self, observation):
        """
        Takes the raw RGB observation from the unwrapped env,
        applies preprocessing + frame‑stack, then returns the
        argmax‑Q action from the online network.
        """
       
        obs = greyscale(observation)
        obs = resize(obs, (84, 84))

        if self.done:
            self.step = 0
            self.done = False
            self.last_action = 0

        if self.step % self.skip == 0:
            if len(self.frame_stack) < 4:
                for _ in range(4):
                    self.frame_stack.append(obs)
            else:
                self.frame_stack.append(obs)

            stacked_frame = torch.stack(list(self.frame_stack), dim=0).unsqueeze(0).to(self.device)
            state = stacked_frame[0].cpu().numpy()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
            # print(f"Simulated State: {state}")
            # print(f"Shape of Simulated State: {state.shape}")

            self.last_action = action_idx
            self.step += 1
            return action_idx
        else:
            self.step += 1
            # epsilon greedy
            if np.random.rand() < 0.01:
                action_idx = self.action_space.sample()
            else:
                action_idx = self.last_action
            self.last_action = action_idx
            return action_idx

if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    agent = Agent()

    for i in range(5):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            if step_count < 0:
                a = env.action_space.sample()
            else:
                a = agent.act(obs)
            obs, r, done, info = env.step(a)
            total_reward += r
            step_count += 1
            # print(f"Step: {step_count}, Action: {a}, Reward: {r}, Done: {done}, Total Reward: {total_reward}")
            env.render()

        print("Finished with reward:", total_reward)