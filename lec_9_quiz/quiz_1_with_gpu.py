import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        done = np.array([x[4] for x in data]).astype(np.float32)

        # NumPy 배열을 PyTorch 텐서로 변환하고 GPU로 이동
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
    def __init__(self, state_size, action_size):
        # Mountain Car 환경에 맞게 수정된 하이퍼파라미터
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # 감가율 증가 (장기적 보상 중요성 증가)
        self.lr = 0.001  # 학습률 증가 (더 빠른 학습)
        self.epsilon = 0.1  # 탐색 확률
        self.buffer_size = 10000
        self.batch_size = 64  # 배치 크기 증가 (GPU 활용도 향상)

        # 네트워크 및 옵티마이저 초기화
        self.qnet = QNet(state_size, action_size).to(device)
        self.qnet_target = QNet(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        # GPU 최적화를 위한 mixed precision 설정
        self.scaler = GradScaler()

        # 타깃 네트워크 초기화
        self.sync_qnet()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.qnet(state_tensor)
            return q_values.cpu().data.numpy().argmax()

    def update(self, state, action, reward, next_state, done):
        # 경험 저장
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        # 배치 데이터 가져오기
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        # Mixed precision 학습
        with autocast():
            # 현재 Q 값 계산
            q_values = self.qnet(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # 타깃 Q 값 계산 (Double DQN 적용)
            with torch.no_grad():
                next_q_values = self.qnet(next_states)
                best_actions = next_q_values.argmax(1)
                next_q_values_target = self.qnet_target(next_states)
                next_q_value = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target = rewards + (1 - dones) * self.gamma * next_q_value

            # 손실 계산
            loss = F.smooth_l1_loss(q_value, target)  # Huber 손실 (안정적인 학습)

        # 역전파 및 가중치 업데이트 (mixed precision 사용)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


# 학습 파라미터 설정
episodes = 300  # 에피소드 수 감소 (GPU 가속으로 충분)
sync_interval = 5  # 타깃 네트워크 동기화 주기 단축
env = gym.make('MountainCar-v0', render_mode='rgb_array')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
reward_history = []
episode_lengths = []


# 학습 진행률 표시 함수
def progress_bar(current, total, bar_length=50):
    percent = current / total
    arrow = '=' * int(bar_length * percent)
    spaces = ' ' * (bar_length - len(arrow))
    print(f"\r[{arrow}{spaces}] {int(percent * 100)}%", end='')


# 에피소드별 학습
print(f"시작: MountainCar-v0, 에피소드: {episodes}, 장치: {device}")
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    step_count = 0
    episode_loss = []

    while not done and step_count < 1000:  # 최대 스텝 제한
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # 수정된 보상 사용
        modified_reward = compute_reward(state, next_state, done)

        # 에이전트 업데이트
        loss = agent.update(state, action, modified_reward, next_state, done)
        if loss is not None:
            episode_loss.append(loss)

        state = next_state
        total_reward += reward  # 원래 보상으로 기록 (성능 비교용)
        step_count += 1

    # 타깃 네트워크 동기화
    if episode % sync_interval == 0:
        agent.sync_qnet()

    # 진행 상황 저장
    reward_history.append(total_reward)
    episode_lengths.append(step_count)

    # 진행 상황 출력
    progress_bar(episode + 1, episodes)
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_history[-10:])
        avg_length = np.mean(episode_lengths[-10:])
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(f"\n에피소드: {episode + 1}/{episodes}, 보상: {total_reward:.2f}, 평균 보상(10): {avg_reward:.2f}, "
              f"스텝: {step_count}, 평균 스텝(10): {avg_length:.2f}, 평균 손실: {avg_loss:.4f}")

# 학습 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(reward_history)
plt.xlabel('에피소드')
plt.ylabel('총 보상')
plt.title('총 보상 추이')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.xlabel('에피소드')
plt.ylabel('에피소드 길이')
plt.title('에피소드 길이 추이')
plt.grid(True)

plt.tight_layout()
plt.savefig('mountain_car_dqn_learning_curve.png')
plt.show()

# 학습된 에이전트 테스트
print("\n학습된 에이전트 테스트 중...")
env_test = gym.make('MountainCar-v0', render_mode='human')
agent.epsilon = 0  # 테스트에서는 탐색 없음 (그리디 정책)

for test_ep in range(3):  # 3번 테스트
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

    print(f"테스트 에피소드 {test_ep + 1}/3, 총 보상: {total_reward:.2f}, 스텝: {step_count}")

print("테스트 완료!")