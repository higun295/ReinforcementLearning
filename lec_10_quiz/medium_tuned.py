import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.preprocessing import StandardScaler

# 환경 설정 및 상태 스케일링
env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = float(env.action_space.low[0])
action_high = float(env.action_space.high[0])

scaler = StandardScaler()
state_samples = np.array([env.observation_space.sample() for _ in range(10000)])
scaler.fit(state_samples)

def scale_state(state):
    state = np.array(state)
    return scaler.transform(state.reshape(1, -1))

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 40),
            nn.ELU(),
            nn.Linear(40, 40),
            nn.ELU()
        )
        self.mu_head = nn.Linear(40, action_dim)
        self.sigma_head = nn.Linear(40, action_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.mu_head(x)
        sigma = torch.nn.functional.softplus(self.sigma_head(x)) + 1e-5
        return mu, sigma

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ELU(),
            nn.Linear(400, 400),
            nn.ELU(),
            nn.Linear(400, 1)
        )

    def forward(self, x):
        return self.net(x)

def train():
    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    actor_optimizer = optim.Adam(policy_net.parameters(), lr=2e-5)
    critic_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

    gamma = 0.99
    episodes = 300
    rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        steps = 0

        while not done:
            state_scaled = torch.FloatTensor(scale_state(state))
            mu, sigma = policy_net(state_scaled)
            dist = torch.distributions.Normal(mu, sigma)
            action_sample = dist.sample()
            action_clipped = torch.clamp(action_sample, action_low, action_high)
            log_prob = dist.log_prob(action_sample).sum(dim=-1)

            # Ensure action shape is (1,) for the environment
            action_np = np.array([action_clipped.item()])
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            steps += 1

            # Critic update
            state_value = value_net(state_scaled)
            next_state_scaled = torch.FloatTensor(scale_state(next_state))
            next_state_value = value_net(next_state_scaled).detach()
            target = reward + gamma * next_state_value
            td_error = target - state_value

            critic_loss = td_error.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor update
            actor_loss = -log_prob * td_error.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode}, Steps: {steps}, Reward: {total_reward:.2f}")

    # 그래프 저장
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PyTorch Actor-Critic on MountainCarContinuous")
    plt.grid()
    plt.savefig("reward_curve.png")
    plt.show()

    return policy_net

def render_and_save_video(policy_net, filename="mountaincar_continuous_actor_critic.mp4"):
    state = env.reset()[0]
    done = False
    frames = []

    while not done:
        state_scaled = torch.FloatTensor(scale_state(state))
        with torch.no_grad():
            mu, sigma = policy_net(state_scaled)
            action = np.array([mu.item()])  # deterministic action
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state

    for _ in range(30):
        frames.append(env.render())

    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved to {filename}")

if __name__ == '__main__':
    trained_policy = train()
    render_and_save_video(trained_policy)
