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

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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

def train(env, qnet, qnet_target, optimizer, episodes, gamma, buffer, batch_size, sync_freq, epsilon, epsilon_decay, epsilon_min):
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

            if steps % sync_freq == 0:
                qnet_target.load_state_dict(qnet.state_dict())

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        total_rewards.append(total_reward)
        if episode % 10 == 0:
            avg = np.mean(total_rewards[-10:])
            print(f"  [Ep {episode}] Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}, Avg(10): {avg:.1f}")

    return total_rewards

def run_experiments():
    state_dim = 2  # MountainCar observation: (position, velocity)
    action_dim = 3  # left, neutral, right

    configs = [
        {"lr": 1e-3, "gamma": 0.99, "batch_size": 64, "epsilon_decay": 0.995},
        {"lr": 5e-4, "gamma": 0.98, "batch_size": 32, "epsilon_decay": 0.990},
        {"lr": 1e-4, "gamma": 0.97, "batch_size": 64, "epsilon_decay": 0.985},
        {"lr": 1e-3, "gamma": 0.95, "batch_size": 32, "epsilon_decay": 0.980},
        {"lr": 5e-4, "gamma": 0.99, "batch_size": 64, "epsilon_decay": 0.975}
    ]

    best_reward = -float('inf')
    best_params = None
    best_policy = None
    all_results = []

    for idx, config in enumerate(configs, 1):
        print(f"\\n[Config {idx}/5] {config}")
        env = gym.make("MountainCar-v0")  # 학습 중에는 렌더링 없음

        qnet = QNet(state_dim, action_dim)
        qnet_target = QNet(state_dim, action_dim)
        qnet_target.load_state_dict(qnet.state_dict())

        optimizer = optim.Adam(qnet.parameters(), lr=config["lr"])
        buffer = ReplayBuffer(10000)

        rewards = train(
            env=env,
            qnet=qnet,
            qnet_target=qnet_target,
            optimizer=optimizer,
            episodes=1000,
            gamma=config["gamma"],
            buffer=buffer,
            batch_size=config["batch_size"],
            sync_freq=100,
            epsilon=1.0,
            epsilon_decay=config["epsilon_decay"],
            epsilon_min=0.01
        )

        all_results.append((config, rewards))
        max_reward = max(rewards)
        if max_reward > best_reward:
            best_reward = max_reward
            best_params = config
            best_policy = qnet

    return all_results, best_params, best_policy

def render_policy(policy_net):
    env = gym.make("MountainCar-v0", render_mode="human")
    state = env.reset()[0]
    done = False

    print("\\n🎥 Best policy 시뮬레이션 시작...")
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = torch.argmax(policy_net(state_tensor)).item()
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        state = next_state

    env.close()
    print("✅ 시뮬레이션 종료.")

# 실행
if __name__ == "__main__":
    results, best_params, best_policy = run_experiments()

    # 보상 그래프
    plt.figure(figsize=(12, 7))
    for config, rewards in results:
        label = f"γ={config['gamma']}, lr={config['lr']}, ε_decay={config['epsilon_decay']}"
        plt.plot(rewards, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("MountainCar DQN - 하이퍼파라미터 비교")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\\n✅ Best Hyperparameters: {best_params}")

    # 최적 정책 렌더링
    render_policy(best_policy)
