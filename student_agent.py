import gym
import torch
import numpy as np
from collections import deque
from pathlib import Path
from torchvision import transforms as T

# make sure your PYTHONPATH lets you import MarioNet from your train.py
from train import MarioNet  

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
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        # ─────────────────────────────────────────────────────────────────

        # build exactly the same transforms as in train.py:
        #   1) permute H×W×C → C×H×W, float tensor
        #   2) grayscale
        #   3) resize to 84×84 (antialias)
        #   4) normalize 0–255 → 0–1
        self.transform = T.Compose([
            T.Lambda(lambda img: torch.tensor(
                np.transpose(img, (2, 0, 1)).copy(),
                dtype=torch.float
            )),
            T.Grayscale(),
            T.Resize((84, 84), antialias=True),
            T.Normalize(0, 255),
        ])

        # a deque to hold our 4-frame stack
        self.frame_stack = deque(maxlen=4)
        # placeholder for the last action (only needed if you wanted to
        # replicate SkipFrame's action‑repeat; you can ignore for testing)
        self.last_action = 0

    def act(self, observation):
        """
        Takes the raw RGB observation from the unwrapped env,
        applies preprocessing + frame‑stack, then returns the
        argmax‑Q action from the online network.
        """
        # 1) preprocess this single raw frame → 1×84×84 tensor
        processed = self.transform(observation).squeeze(0)

        # 2) initialize or update our 4‑frame buffer
        if len(self.frame_stack) < 4:
            # on cold start, fill with the same first frame
            for _ in range(4):
                self.frame_stack.append(processed)
        else:
            self.frame_stack.append(processed)

        # 3) build a [1,4,84,84] batch and run the network
        state = torch.stack(list(self.frame_stack), dim=0) \
                     .unsqueeze(0) \
                     .to(self.device)

        with torch.no_grad():
            q_vals = self.net(state, model="online")

        action = torch.argmax(q_vals, dim=1).item()
        return action

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT 

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

agent = Agent()
obs = env.reset()
done = False
total_reward = 0

while not done:
    a = agent.act(obs)
    obs, r, done, info = env.step(a)
    total_reward += r
    env.render()

print("Finished with reward:", total_reward)
