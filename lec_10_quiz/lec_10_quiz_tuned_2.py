import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 환경 및 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Policy Network
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return torch.softmax(x, dim=-1)

# Value Network
class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 학습 함수
def train(config):
    print(f"\n[실험 시작] {config['name']}")
    policy_net = PolicyNet(state_dim, action_dim, config['hidden']).to(device)
    value_net = ValueNet(state_dim, config['hidden']).to(device)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=config['policy_lr'])
    optimizer_value = optim.Adam(value_net.parameters(), lr=config['value_lr'])

    gamma = config['gamma']
    entropy_coef = config['entropy_coef']
    episodes = 500
    rewards = []

    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = policy_net(s)
            if torch.isnan(probs).any():
                print(f"[Ep {ep}] NaN 발생! 스킵")
                break
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            velocity = next_state[1]
            reward += abs(velocity) * config['v_coeff']
            reward += (next_state[0] + 0.5) * config['p_coeff']

            s_next = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            target = reward + gamma * value_net(s_next) * (0 if done else 1)
            value = value_net(s)
            td_error = target - value

            value_loss = td_error.pow(2).mean()
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            log_prob = torch.log(probs[0, action])
            entropy = dist.entropy().mean()
            policy_loss = -log_prob * td_error.detach() - entropy_coef * entropy
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if ep % 50 == 0:
            print(f"[Ep {ep}] Reward: {total_reward:.1f}, Avg(10): {np.mean(rewards[-10:]):.2f}")

    return rewards

# 실험 설정 리스트
config_list = [
    {
        'name': 'baseline',
        'hidden': [128, 128, 64],
        'policy_lr': 0.0003,
        'value_lr': 0.001,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'v_coeff': 2,
        'p_coeff': 1,
    },
    {
        'name': 'low_entropy',
        'hidden': [128, 128, 64],
        'policy_lr': 0.0003,
        'value_lr': 0.001,
        'gamma': 0.99,
        'entropy_coef': 0.001,
        'v_coeff': 2,
        'p_coeff': 1,
    },
    {
        'name': 'deeper_net',
        'hidden': [256, 128, 64],
        'policy_lr': 0.0003,
        'value_lr': 0.001,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'v_coeff': 2,
        'p_coeff': 1,
    },
    {
        'name': 'more_shaping',
        'hidden': [128, 128, 64],
        'policy_lr': 0.0003,
        'value_lr': 0.001,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'v_coeff': 3,
        'p_coeff': 2,
    }
]

# 실험 실행
all_rewards = {}
for config in config_list:
    rewards = train(config)
    all_rewards[config['name']] = rewards

# 그래프 출력
plt.figure(figsize=(10, 6))
for name, rewards in all_rewards.items():
    plt.plot(rewards, label=name)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Actor-Critic Multi-Config Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("multi_config_comparison.png")
plt.show()
