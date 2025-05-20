import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

# 환경 설정
env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = float(env.action_space.low[0])
action_high = float(env.action_space.high[0])

# 네트워크 정의
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu = nn.Linear(32, action_dim)
        self.sigma = nn.Linear(32, action_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        sigma = torch.clamp(torch.nn.functional.softplus(self.sigma(x)), min=1e-4)
        return mu, sigma

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# 학습 함수
def train():
    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

    gamma = 0.99
    episodes = 500
    rewards = []

    for episode in range(episodes):
        _ = env.reset()
        env.state = np.array([-0.4, 0.0])  # 초기 상태 강제 설정
        state = env.state

        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mu, sigma = policy_net(state_tensor)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action_clipped = torch.clamp(action, action_low, action_high)
            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy().flatten())
            done = terminated or truncated

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            value = value_net(state_tensor)
            next_value = value_net(next_state_tensor).detach()
            target = reward + gamma * next_value
            delta = target - value

            # Critic update
            loss_value = delta.pow(2).mean()
            optimizer_value.zero_grad()
            loss_value.backward()
            optimizer_value.step()

            # Actor update
            loss_policy = -log_prob * delta.detach()
            optimizer_policy.zero_grad()
            loss_policy.backward()
            optimizer_policy.step()

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")

    # 보상 그래프 저장
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Actor-Critic on MountainCarContinuous")
    plt.grid()
    plt.savefig("reward_curve.png")
    plt.show()

    return policy_net

# 영상 저장 함수
def render_and_save_video(policy_net, filename="mountaincar_actor_critic__1.mp4"):
    _ = env.reset()
    env.state = np.array([-0.4, 0.0])
    state = env.state
    done = False
    frames = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, _ = policy_net(state_tensor)
        action = mu.detach().numpy().flatten()
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state

    for _ in range(30):
        frames.append(env.render())

    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved to {filename}")

# 실행
if __name__ == '__main__':
    policy = train()
    render_and_save_video(policy)