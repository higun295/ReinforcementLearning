# Actor-Critic 튜닝 버전 (안정화된): entropy, advantage normalization, NaN 방지 포함

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MountainCar 환경 초기화
env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]  # 2
action_dim = env.action_space.n             # 3

# Policy Network (Actor)
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[128, 128, 64]):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.out = nn.Linear(hidden[2], output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.out(x), dim=-1)

# Value Network (Critic)
class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden=[128, 128, 64]):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.out = nn.Linear(hidden[2], 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

# 네트워크 및 옵티마이저 초기화
policy_net = PolicyNet(state_dim, action_dim).to(device)
value_net = ValueNet(state_dim).to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.0003)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.001)

# 하이퍼파라미터 설정
gamma = 0.99
episodes = 1000
entropy_coef = 0.01

# 학습 루프
total_rewards = []

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = policy_net(state_tensor)

        # NaN 체크 (softmax 안정성 확보)
        if torch.isnan(probs).any():
            print(f"[Ep {episode}] NaN detected in policy output! Skipping update.")
            break

        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # reward shaping: 속도 + 위치 보상 (과도하지 않게 조절)
        velocity = next_state[1]
        reward += abs(velocity) * 2
        reward += (next_state[0] + 0.5) * 1

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        # TD target과 advantage 계산
        target = reward + gamma * value_net(next_state_tensor) * (0 if done else 1)
        value = value_net(state_tensor)
        advantage = (target - value).detach()

        # advantage 정규화 (안정성 추가)
        if advantage.std() > 1e-6:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Critic 업데이트
        value_loss = (target - value).pow(2).mean()
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # Actor 업데이트: entropy regularization 포함
        log_prob = torch.log(probs[0, action])
        entropy = dist.entropy().mean()
        policy_loss = -log_prob * advantage - entropy_coef * entropy
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        state = next_state
        total_reward += reward

    total_rewards.append(total_reward)
    if episode % 50 == 0:
        avg_reward = np.mean(total_rewards[-10:])
        print(f"[Ep {episode}] Total reward: {total_reward:.1f}, Avg(10): {avg_reward:.2f}")

# 보상 그래프 출력
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Stable Actor-Critic on MountainCar")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_plot_stable.png")
plt.show()

# 영상 저장 함수
def render_and_save_video(policy_net, filename="mountaincar_actor_critic_stable.mp4"):
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state = env.reset()[0]
    frames = []
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = policy_net(state_tensor)
        action = torch.argmax(probs).item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())

    for _ in range(30):
        frames.append(env.render())

    env.close()
    imageio.mimsave(filename, frames, fps=30)

render_and_save_video(policy_net)
