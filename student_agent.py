import gym
import torch
import numpy as np
from pathlib import Path
from collections import deque

from torchvision import transforms as T

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# import your network definition
from train import MarioNet

class Agent(object):
    """Takes raw RGB frames, applies the same preprocessing as in training,
    stacks 4 frames, and returns the greedy DQN action."""
    def __init__(self):
        # action dim
        self.action_space = gym.spaces.Discrete(12)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build net and load weights
        self.net = MarioNet(input_dim=(4, 84, 84), output_dim=self.action_space.n).to(self.device)
        # ckpt_path = self._latest_ckpt(Path("checkpoints"))
        ckpt = "./mario_net_23.chkpt"
        checkpoint = torch.load(ckpt, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.net.eval()
        print(f"[Agent] Loaded checkpoint: {ckpt}")

        # frame buffer & preprocess pipeline
        self.buffer = deque(maxlen=4)
        self.transform = T.Compose([
            T.ToPILImage(),                 # from H×W×C array
            T.Grayscale(num_output_channels=1),
            T.Resize((84, 84), antialias=True),
            T.ToTensor(),                   # scales to [0,1] and gives shape 1×84×84
        ])

    def _latest_ckpt(self, base_dir: Path) -> Path:
        """Finds the highest‐numbered mario_net_*.chkpt under the most recent run dir."""
        runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No run folders in {base_dir}")
        latest = runs[-1]
        files = sorted(latest.glob("mario_net_*.chkpt"),
                       key=lambda f: int(f.stem.split("_")[-1]))
        if not files:
            raise FileNotFoundError(f"No .chkpt files in {latest}")
        return files[-1]

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Grayscale, resize, normalize, returns a 84×84 float32 array."""
        # transform → torch tensor 1×84×84
        t = self.transform(frame)  
        # squeeze channel, to numpy (84×84)
        return t.squeeze(0).cpu().numpy()

    def act(self, observation):
        # raw env returns a (240×256×3) uint8 array
        frame = observation if isinstance(observation, np.ndarray) else observation[0]

        # preprocess & push into buffer
        proc = self._preprocess(frame)
        self.buffer.append(proc)

        # at start, pad with repeats of the first frame
        if len(self.buffer) < 4:
            while len(self.buffer) < 4:
                self.buffer.append(proc)

        # stack into (4×84×84)
        state = np.stack(self.buffer, axis=0)

        # ensure C‐contiguous
        state = np.ascontiguousarray(state)

        # make batch 1×4×84×84
        tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # forward & pick greedy
        with torch.no_grad():
            q = self.net(tensor, model="online")
            return int(q.argmax(dim=1).item())


def main():
    # 1) build *raw* env (no grayscale/resize/stack wrappers here)
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # 2) agent (loads model & sets up its own preproc)
    agent = Agent()

    # 3) play one episode
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    print(f"\n=== Episode finished! Total accumulated reward: {total_reward:.2f} ===")
    env.close()


if __name__ == "__main__":
    main()
