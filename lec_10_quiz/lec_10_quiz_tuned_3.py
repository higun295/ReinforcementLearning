# 개선된 Actor-Critic: 보상 구조 조정 + 네트워크 아키텍처 확장 + 결과 그래프 및 영상 저장 포함

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

# 환경 초기화
env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 개선된 Policy Network
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

# 개선된 Value Network
class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# 네트워크 및 옵티마이저 초기화
policy_net = PolicyNet(state_dim, action_dim).to(device)
value_net = ValueNet(state_dim).to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.0005)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.001)

# 하이퍼파라미터
gamma = 0.99
entropy_coef = 0.005
episodes = 500

# 학습 루프
total_rewards = []

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = policy_net(state_tensor)

        if torch.isnan(probs).any():
            print(f"[Ep {episode}] NaN in policy output. Skipping.")
            break

        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 보상 구조 개선: 위치 기반 shaping + 도달 시 보너스
        position = next_state[0]
        velocity = next_state[1]

        # 기본 보상
        base_reward = -1

        # shaping 보상 추가
        shaping_reward = 10 * (position + 0.5) + 5 * abs(velocity)

        # 목표 지점 도달 시 추가 보너스
        if position >= 0.5:
            shaping_reward += 100

        total_step_reward = base_reward + shaping_reward

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        target = total_step_reward + gamma * value_net(next_state_tensor) * (0 if done else 1)
        value = value_net(state_tensor)
        td_error = target - value

        # Critic update
        value_loss = td_error.pow(2).mean()
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # Actor update
        log_prob = torch.log(probs[0, action])
        entropy = dist.entropy().mean()
        policy_loss = -log_prob * td_error.detach() - entropy_coef * entropy
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        state = next_state
        total_reward += total_step_reward

    total_rewards.append(total_reward)
    if episode % 50 == 0:
        avg = np.mean(total_rewards[-10:])
        print(f"[Ep {episode}] Total reward: {total_reward:.1f}, Avg(10): {avg:.2f}")

# 보상 그래프
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Improved Actor-Critic on MountainCar")
plt.grid(True)
plt.tight_layout()
plt.savefig("improved_reward_plot.png")
plt.show()

# 영상 저장 함수
def render_and_save_video(policy_net, filename="mountaincar_improved.mp4"):
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