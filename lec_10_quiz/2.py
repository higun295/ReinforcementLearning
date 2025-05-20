import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

class PolicyNet(nn.Module):  # Actor
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):  # Critic
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

def train():
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=5e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=5e-4)

    gamma = 0.99
    episodes = 500
    rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        log_probs = []
        values = []
        rewards_ep = []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value = value_net(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # --- Reward shaping ---
            reward += abs(next_state[1]) * 2  # 속도 기반 보상
            reward += (next_state[0] + 0.5) * 1  # 위치 기반 보상

            log_probs.append(log_prob)
            values.append(value)
            rewards_ep.append(reward)

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # compute returns and losses
        returns = []
        G = 0
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)

        advantages = returns - values.detach()

        loss_policy = -(log_probs * advantages).mean()
        loss_value = nn.SmoothL1Loss()(values, returns)  # Huber Loss

        optimizer_policy.zero_grad()
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer_policy.step()

        optimizer_value.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        optimizer_value.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, reward: {total_reward:.2f}")

    # --- 그래프 저장 ---
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Actor-Critic on MountainCar (Deep Wide Network)")
    plt.grid()
    plt.savefig("reward_curve.png")  # 그래프 저장
    plt.show()

    return policy_net, rewards

def render_and_save_video(policy_net, filename="actor_critic_mountaincar.mp4"):
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state = env.reset()[0]
    frames = []
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = policy_net(state_tensor)
        action = torch.argmax(probs).item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())

    for _ in range(30):
        frames.append(env.render())

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved to {filename}")

if __name__ == '__main__':
    policy, rewards = train()
    render_and_save_video(policy)
