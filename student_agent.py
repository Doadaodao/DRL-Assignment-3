import gym
import torch
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image
from torchvision import transforms as T

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack

# import network & wrappers from your training file
from train import MarioNet, SkipFrame, GrayScaleObservation, ResizeObservation

class Agent(object):
    """Agent that loads a trained DQN and returns the argmax action."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        # pick device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # build network
        self.net = MarioNet(input_dim=(4, 84, 84), output_dim=self.action_space.n).to(self.device)
        # locate & load the latest checkpoint
        # ckpt_path = self._latest_ckpt(Path("checkpoints"))
        ckpt_path = "./mario_net_23.chkpt"
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.net.eval()
        print(f"Loaded checkpoint: {ckpt_path}")

        # preprocessing pipeline: RGB→PIL→grayscale(1ch)→84×84→Tensor([0,1])
        self.transform = T.Compose([
            T.ToPILImage(),  
            T.Grayscale(num_output_channels=1),
            T.Resize((84, 84), antialias=True),
            T.ToTensor(),    # gives float32 in [0,1]
        ])
        # frame buffer for last 4 preprocess frames
        self.frame_buffer = deque(maxlen=4)

    def _latest_ckpt(self, base_dir: Path) -> Path:
        """Walks checkpoints/<run‑timestamp>/*.chkpt and returns the highest-numbered file."""
        runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No run folders in {base_dir}")
        latest_run = runs[-1]
        files = list(latest_run.glob("mario_net_*.chkpt"))
        if not files:
            raise FileNotFoundError(f"No .chkpt files in {latest_run}")
        files.sort(key=lambda f: int(f.stem.split("_")[-1]))
        return files[-1]

    def act(self, raw_rgb: np.ndarray) -> int:
        """
        raw_rgb: HxWx3 uint8 frame from the unwrapped env.
        Returns: int action in [0, 11].
        """
        # 1) make sure we have a C‑contiguous array
        frame_np = np.ascontiguousarray(raw_rgb)

        # 2) grayscale + resize + to‐tensor([0,1])
        frame_t  = self.transform(frame_np)       # shape [1,84,84]

        # 3) fill or append to our 4‑frame buffer
        if len(self.frame_buffer) < 4:
            # on very first call, replicate the first frame
            while len(self.frame_buffer) < 4:
                self.frame_buffer.append(frame_t)
        self.frame_buffer.append(frame_t)

        # 4) stack into a single tensor [1,4,84,84]
        state = torch.cat(list(self.frame_buffer), dim=0) \
                     .unsqueeze(0) \
                     .to(self.device)   # [1,4,84,84]

        # 5) forward + greedy
        with torch.no_grad():
            q_vals = self.net(state, model="online")
            action = int(q_vals.argmax(dim=1).item())

        return action

def main():
    # only wrap for discrete action set (and optionally skip frames)
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)

    agent = Agent(checkpoint_dir="checkpoints")

    obs = env.reset()   # this is raw RGB now
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Total accumulated reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
