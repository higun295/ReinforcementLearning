import copy
import matplotlib.pyplot as plt
import numpy as np
import gym
import random
from collections import deque
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in batch])
        action = np.array([x[1] for x in batch])
        reward = np.array([x[2] for x in batch])
        next_state = np.stack([x[3] for x in batch])
        done = np.array([x[4] for x in batch]).astype(np.int32)
        return state, action, reward, next_state, done


class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


def compute_reward(state, next_state, done):
    if done and next_state[0] >= 0.5:
        return 100.0
    height_reward = next_state[0] + 0.5
    velocity_reward = abs(next_state[1]) * 10 if next_state[1] > 0 else 0
    return -1.0 + height_reward + velocity_reward


class DQNAgent:
    def __init__(self, gamma, lr, epsilon, buffer_size, batch_size, action_size):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_size = action_size

        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)


# 실험 설정
hyperparam_configs = [
    {"gamma": 0.99, "lr": 0.001, "epsilon": 0.1},
    {"gamma": 0.98, "lr": 0.0005, "epsilon": 0.2},
    {"gamma": 0.97, "lr": 0.0001, "epsilon": 0.05},
    {"gamma": 0.99, "lr": 0.0005, "epsilon": 0.3},
    {"gamma": 0.95, "lr": 0.001, "epsilon": 0.15},
]

results = []

for idx, config in enumerate(hyperparam_configs, 1):
    print(f"\n[실험 {idx}] config: {config}")
    env = gym.make("MountainCar-v0")
    agent = DQNAgent(
        gamma=config["gamma"],
        lr=config["lr"],
        epsilon=config["epsilon"],
        buffer_size=10000,
        batch_size=32,
        action_size=3
    )

    episodes = 300
    sync_interval = 10
    reward_history = []

    for ep in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 1000:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            mod_reward = compute_reward(state, next_state, done)

            agent.update(state, action, mod_reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        if ep % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)
        if ep % 10 == 0:
            print(f"EP {ep} | Total Reward: {total_reward:.1f}")

    results.append((config, reward_history))

# 그래프 출력
plt.figure(figsize=(12, 7))
for config, history in results:
    label = f"γ={config['gamma']}, lr={config['lr']}, ε={config['epsilon']}"
    plt.plot(history, label=label)
plt.title("Hyperparameter 별 에피소드 보상 비교")
plt.xlabel("에피소드")
plt.ylabel("총 보상")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
