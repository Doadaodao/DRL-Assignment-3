import gym
import torch
import numpy as np
from pathlib import Path

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MarioNet(input_dim=(4, 84, 84), output_dim=self.action_space.n).to(self.device)

        ckpt_path = self._latest_ckpt(Path("checkpoints"))
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.net.eval()
        print(f"Loaded checkpoint: {ckpt_path}")

    def _latest_ckpt(self, base_dir: Path) -> Path:
        runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No run folders in {base_dir}")
        latest_run = runs[-1]
        files = sorted(latest_run.glob("mario_net_*.chkpt"),
                       key=lambda f: int(f.stem.split("_")[-1]))
        if not files:
            raise FileNotFoundError(f"No .chkpt files in {latest_run}")
        return files[-1]

    def act(self, observation):
        """
        Preprocess the incoming observation into a contiguous float32 tensor
        and return the greedy action.
        """
        # 1) pull out the raw (4×84×84) array
        if hasattr(observation, "__array__"):
            obs = observation.__array__()
        else:
            obs = np.array(observation)

        # 2) ensure positive strides / C‑contiguous memory
        obs = np.ascontiguousarray(obs)

        # 3) convert to torch tensor and batch it
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 4) forward & argmax
        with torch.no_grad():
            q_vals = self.net(state, model="online")
            return int(q_vals.argmax(dim=1).item())

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    agent = Agent()

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    print(f"\n=== Episode finished! Total accumulated reward: {total_reward:.2f} ===")
    env.close()

if __name__ == "__main__":
    main()
