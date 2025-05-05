import gym
import torch
import numpy as np
from collections import deque
from torchvision import transforms as T
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from agent import MarioNet

def transform(observation):
    def permute_orientation(observation):
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
    
    obs = greyscale(observation)
    obs = resize(obs, (84, 84))
    return obs

class Agent(object):
    """Loads a trained MarioNet checkpoint and applies the same
    SkipFrame/GrayScale/Resize/FrameStack pipeline in-act."""
    def __init__(self):
        # action space must match JoypadSpace(COMPLEX_MOVEMENT)
        self.action_space = gym.spaces.Discrete(12)

        # device & model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MarioNet(input_dim=(4, 84, 84), output_dim=12).to(self.device)

        # load checkpoint
        ckpt_path = "./checkpoints/2025-04-29T00-48-10/mario_net_55.chkpt"
        ckpt_path = "./checkpoints/2025-05-04T19-16-53/mario_net_8.chkpt"
        ckpt_path = "./mario_net_5.chkpt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()

        self.step = 0
        self.skip = 4
        self.last_action = 0
        self.done = False

        self.epsilon = 0.0001

        self.frame_stack = deque(maxlen=4)
        

    def act(self, observation):
        """
        Takes the raw RGB observation from the unwrapped env,
        applies preprocessing + frame‑stack, then returns the
        argmax‑Q action from the online network.
        """
       
        obs = transform(observation)

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

            self.last_action = action_idx
            self.step += 1

            # epsilon greedy
            if np.random.rand() < self.epsilon:
                action_idx = self.action_space.sample()
            return action_idx
        else:
            self.step += 1
            return self.last_action

if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    scores = []
    for i in range(5):
        agent = Agent()
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            if step_count < 4: 
                a = env.action_space.sample()
            else:
                a = agent.act(obs)
            obs, r, done, info = env.step(a)
            total_reward += r
            step_count += 1
            # print(f"Step: {step_count}, Action: {a}, Reward: {r}, Done: {done}, Total Reward: {total_reward}")
            # env.render()
        scores.append(total_reward)

        print("Finished with reward:", total_reward)
    print("Average score:", np.mean(scores))