import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import imageio

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(QNet, self).__init__()
        self.layers = nn.ModuleList()
        input_dim = state_dim
        for h in hidden_units:
            self.layers.append(nn.Linear(input_dim, h))
            input_dim = h
        self.out = nn.Linear(input_dim, action_dim)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def train(env, qnet, qnet_target, optimizer, gamma, buffer, batch_size, sync_interval, epsilon, episodes=500):
    total_rewards = []
    steps = 0

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        while True:
            steps += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = qnet(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                s, a, r, s_, d = buffer.sample(batch_size)

                s = torch.FloatTensor(s)
                a = torch.LongTensor(a).unsqueeze(1)
                r = torch.FloatTensor(r).unsqueeze(1)
                s_ = torch.FloatTensor(s_)
                d = torch.FloatTensor(d).unsqueeze(1)

                q_values = qnet(s).gather(1, a)
                next_q_values = qnet_target(s_).max(1)[0].detach().unsqueeze(1)
                target = r + gamma * next_q_values * (1 - d)

                loss = nn.MSELoss()(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % sync_interval == 0:
                qnet_target.load_state_dict(qnet.state_dict())

            if done:
                break

        total_rewards.append(total_reward)
        if episode % 50 == 0:
            avg = np.mean(total_rewards[-10:])
            print(f"[Ep {episode}] Reward: {total_reward:.1f}, Avg(10): {avg:.1f}")

    return total_rewards

def render_and_save_video(policy_net, filename="mountaincar_single_run_64_32.mp4"):
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state = env.reset()[0]
    done = False
    frames = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = torch.argmax(policy_net(state_tensor)).item()
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state

    for _ in range(30):
        frames.append(env.render())

    env.close()
    imageio.mimsave(filename, frames, fps=30)

# 실행
if __name__ == "__main__":
    config = {
        "hidden_units": [64, 32],
        "epsilon": 0.001,
        "gamma": 0.99,
        "lr": 0.001,
        "episodes": 500,
        "buffer_size": 10000,
        "batch_size": 64,
        "sync_interval": 100
    }

    for k, v in config.items():
        print(f"  {k}: {v}")

    env = gym.make("MountainCar-v0")
    qnet = QNet(2, 3, config["hidden_units"])
    qnet_target = QNet(2, 3, config["hidden_units"])
    qnet_target.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=config["lr"])
    buffer = ReplayBuffer(config["buffer_size"])

    rewards = train(
        env=env,
        qnet=qnet,
        qnet_target=qnet_target,
        optimizer=optimizer,
        gamma=config["gamma"],
        buffer=buffer,
        batch_size=config["batch_size"],
        sync_interval=config["sync_interval"],
        epsilon=config["epsilon"],
        episodes=config["episodes"]
    )

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("MountainCar")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    render_and_save_video(qnet)
