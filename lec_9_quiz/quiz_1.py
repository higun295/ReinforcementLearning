import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
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
        x = self.l3(x)
        return x


# 수정된 보상 함수 (학습 가속화를 위한 보상 설계)
def compute_reward(state, next_state, done):
    """Mountain Car 환경의 보상 함수를 수정하여 학습 효율성 개선"""
    original_reward = -1.0  # 원래 보상: 매 단계마다 -1

    # 목표 도달 시 큰 보상
    if done and next_state[0] >= 0.5:
        return 100.0

    # 높이에 기반한 보상
    height_reward = next_state[0] + 0.5  # 높이에 비례한 보상 (-0.5 ~ +0.5)

    # 속도에 기반한 보상 (오른쪽으로 빠를수록 좋음)
    velocity_reward = abs(next_state[1]) * 10 if next_state[1] > 0 else 0

    # 최종 보상: 기본 보상 + 높이 보상 + 속도 보상
    return original_reward + height_reward + velocity_reward


class DQNAgent:
    def __init__(self):
        # Mountain Car 환경에 맞게 수정된 하이퍼파라미터
        self.gamma = 0.99  # 감가율 증가 (장기적 보상 중요성 증가)
        self.lr = 0.001  # 학습률 증가 (더 빠른 학습)
        self.epsilon = 0.1  # 탐색 확률
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 3  # Mountain Car: 왼쪽, 정지, 오른쪽 (0, 1, 2)

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
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
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
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


# 학습 파라미터 설정
episodes = 1000  # 에피소드 수 증가 (Mountain Car는 학습이 더 어려움)
sync_interval = 10  # 타깃 네트워크 동기화 주기
env = gym.make('MountainCar-v0', render_mode='rgb_array')  # Mountain Car 환경
agent = DQNAgent()
reward_history = []

# 에피소드별 학습
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < 1000:  # 최대 스텝 제한
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # 수정된 보상 사용
        modified_reward = compute_reward(state, next_state, done)

        agent.update(state, action, modified_reward, next_state, done)
        state = next_state
        total_reward += reward  # 원래 보상으로 기록 (성능 비교용)
        step_count += 1

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)

    # 학습 과정 출력 (10 에피소드마다)
    if episode % 10 == 0:
        avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
        print(f"에피소드: {episode}, 총 보상: {total_reward:.2f}, 평균 보상(최근 10개): {avg_reward:.2f}, 스텝: {step_count}")

# 학습 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(reward_history)
plt.xlabel('에피소드')
plt.ylabel('총 보상')
plt.title('Mountain Car DQN 학습 곡선')
plt.grid(True)
plt.savefig('mountain_car_dqn_learning_curve.png')
plt.show()

# 학습된 에이전트 테스트
print("\n학습된 에이전트 테스트 중...")
env_test = gym.make('MountainCar-v0', render_mode='human')
agent.epsilon = 0  # 테스트에서는 탐색 없음 (그리디 정책)

for test_ep in range(5):  # 5번 테스트
    state = env_test.reset()[0]
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < 1000:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env_test.step(action)
        done = terminated | truncated
        state = next_state
        total_reward += reward
        step_count += 1
        env_test.render()

    print(f"테스트 에피소드 {test_ep + 1}/5, 총 보상: {total_reward:.2f}, 스텝: {step_count}")

print("테스트 완료!")