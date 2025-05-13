import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

        # NumPy 배열을 PyTorch 텐서로 변환
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        return state, action, reward, next_state, done


# Dueling DQN 네트워크 구현
class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNet, self).__init__()

        # 공통 특성 추출 레이어
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 가치 스트림 (V)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 장점 스트림 (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - 평균(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


# 강화된 보상 함수
def compute_reward(state, next_state, done, action=None):
    # 목표 도달 시 매우 큰 보상
    if done and next_state[0] >= 0.5:
        return 200.0

    # 높이 기반 보상 (더 가파르게)
    position_reward = (next_state[0] - (-1.2)) / (0.6 - (-1.2))  # -1.2~0.6 범위를 0~1로 정규화
    height_reward = position_reward * 15  # 높이에 더 큰 가중치

    # 속도 기반 보상 (방향성 고려)
    velocity_reward = 0
    # 산을 오르기 위한 진자 움직임 장려
    if (next_state[1] > 0 and next_state[0] > -0.5) or (next_state[1] < 0 and next_state[0] < -0.4):
        velocity_reward = abs(next_state[1]) * 30

    # 진전 보상 (이전 위치보다 더 높이 올라갔을 때)
    progress_reward = 0
    if next_state[0] > state[0]:
        progress_reward = (next_state[0] - state[0]) * 50

    # 최종 보상
    return -1 + height_reward + velocity_reward + progress_reward


class DQNAgent:
    def __init__(self):
        # Mountain Car 환경을 위한 설정
        env = gym.make('MountainCar-v0')
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        env.close()

        # 하이퍼파라미터
        self.gamma = 0.99  # 감마 유지
        self.lr = 0.0005  # 더 작은 학습률
        self.epsilon_start = 1.0  # 초기 입실론
        self.epsilon_end = 0.01  # 최종 입실론
        self.epsilon_decay = 0.995  # 입실론 감소율
        self.epsilon = self.epsilon_start
        self.buffer_size = 100000  # 버퍼 크기 증가
        self.batch_size = 64  # 배치 크기 증가
        self.update_count = 0  # 업데이트 카운터 추적

        # 네트워크 및 옵티마이저
        self.qnet = DuelingQNet(self.state_size, self.action_size).to(device)
        self.qnet_target = DuelingQNet(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

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

        # 현재 Q 값 계산
        q_values = self.qnet(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 타깃 Q 값 계산 (Double DQN)
        with torch.no_grad():
            # 행동 선택은 현재 네트워크로
            best_actions = self.qnet(next_states).argmax(1, keepdim=True)
            # Q 값 계산은 타깃 네트워크로
            next_q_values = self.qnet_target(next_states).gather(1, best_actions).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q_values

        # Huber Loss 사용 (더 안정적인 학습)
        loss = F.smooth_l1_loss(q_value, target)

        # 옵티마이저 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 (안정적인 학습)
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
        self.optimizer.step()

        # 입실론 감소
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 업데이트 카운터 증가
        self.update_count += 1

        return loss.item()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


# 학습 결과를 저장하는 함수
def save_results(reward_history, filename="mountain_car_results"):
    # 그래프 저장
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history)
    plt.xlabel('에피소드')
    plt.ylabel('총 보상')
    plt.title('Mountain Car DQN 학습 곡선')
    plt.grid(True)
    plt.savefig(f'{filename}.png')

    # 10개 에피소드 단위로 이동 평균 추가
    window_size = 10
    avg_rewards = []
    for i in range(len(reward_history) - window_size + 1):
        avg_rewards.append(np.mean(reward_history[i:i + window_size]))

    plt.figure(figsize=(12, 6))
    plt.plot(range(window_size - 1, len(reward_history)), avg_rewards)
    plt.xlabel('에피소드')
    plt.ylabel(f'{window_size}개 에피소드 평균 보상')
    plt.title(f'Mountain Car DQN 이동 평균 (윈도우 크기: {window_size})')
    plt.grid(True)
    plt.savefig(f'{filename}_moving_avg.png')

    # 학습 결과 저장
    import pickle
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump({
            'reward_history': reward_history,
            'avg_rewards': avg_rewards
        }, f)


# 메인 학습 및 테스트 코드
def main():
    # 환경 및 에이전트 초기화
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    agent = DQNAgent()

    # 학습 파라미터
    episodes = 500
    sync_interval = 5
    reward_history = []
    best_reward = -float('inf')
    best_model_path = 'best_mountain_car_model.pth'

    # 경험 버퍼 초기화 (랜덤 데이터로)
    print("=== 경험 버퍼 초기화 중 ===")
    state = env.reset()[0]
    for _ in range(1000):  # 1000개의 초기 경험 수집
        action = np.random.randint(0, agent.action_size)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        modified_reward = compute_reward(state, next_state, done, action)
        agent.replay_buffer.add(state, action, modified_reward, next_state, done)
        if done:
            state = env.reset()[0]
        else:
            state = next_state

    # 학습 시작
    print("=== 학습 시작 ===")
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 1000:  # 최대 1000 스텝
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 수정된 보상 사용
            modified_reward = compute_reward(state, next_state, done, action)

            # 에이전트 업데이트
            agent.update(state, action, modified_reward, next_state, done)

            state = next_state
            total_reward += reward  # 원래 보상으로 기록
            step_count += 1

        # 타깃 네트워크 동기화
        if episode % sync_interval == 0:
            agent.sync_qnet()

        # 진행 상황 기록
        reward_history.append(total_reward)

        # 최고 성능 모델 저장
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.qnet.state_dict(), best_model_path)
            print(f"새로운 최고 성능 모델 저장! 보상: {best_reward:.2f}")

        # 진행 상황 출력
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(
                f"에피소드: {episode + 1}/{episodes}, 보상: {total_reward:.2f}, 평균 보상(10): {avg_reward:.2f}, 스텝: {step_count}, 입실론: {agent.epsilon:.4f}")

    # 학습 종료
    env.close()

    # 학습 결과 저장
    save_results(reward_history)

    # 학습된 에이전트로 테스트
    print("\n=== 최종 모델로 테스트 ===")
    agent.qnet.load_state_dict(torch.load(best_model_path))
    test_agent(agent)


def test_agent(agent):
    print("\n=== 학습된 에이전트 테스트 ===")
    env = gym.make('MountainCar-v0', render_mode='human')

    # 탐색 없이 테스트
    agent.epsilon = 0

    for test_ep in range(5):  # 5번 테스트
        state = env.reset()[0]
        done = False
        total_reward = 0
        step_count = 0

        while not done and step_count < 1000:
            # 행동 선택
            action = agent.get_action(state)

            # 환경 진행
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 상태 및 보상 업데이트
            state = next_state
            total_reward += reward
            step_count += 1

        print(f"테스트 {test_ep + 1}/5, 총 보상: {total_reward:.2f}, 스텝: {step_count}")

    env.close()
    print("테스트 완료!")


if __name__ == "__main__":
    main()