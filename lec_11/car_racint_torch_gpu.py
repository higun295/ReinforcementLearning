from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import time
import imageio


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


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(64 * 64, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        return x


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.feature = CNNFeatureExtractor()
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.feature(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self, device='cpu'):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 5
        self.device = device

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).to(self.device)
        self.qnet_target = QNet(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                qs = self.qnet(state)
            return qs.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        qs = self.qnet(state)
        q = qs[torch.arange(self.batch_size), action]

        with torch.no_grad():
            next_qs = self.qnet_target(next_state)
            next_q = next_qs.max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


def preprocess(state):
    state = np.transpose(state, (2, 0, 1))
    state = state / 255.0
    return state


# 100회 이동평균 계산 함수 추가
def calculate_moving_average(data, window_size=100):
    moving_averages = []
    for i in range(len(data)):
        if i < window_size - 1:
            avg = np.mean(data[:i + 1])
        else:
            avg = np.mean(data[i - window_size + 1:i + 1])
        moving_averages.append(avg)
    return moving_averages


# 영상 저장 함수 추가
def save_gameplay_video(agent, filename="racing_gameplay.mp4", max_frames=1000):
    """학습된 에이전트의 게임플레이를 비디오로 기록"""
    # 비디오 기록용 환경 생성
    env_video = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')

    state = env_video.reset()[0]
    state = preprocess(state)
    done = False
    frames = []
    total_reward = 0
    steps = 0

    agent.epsilon = 0  # 탐험 끄기

    while not done and steps < max_frames:
        # 행동 선택 (학습된 정책 사용)
        action = agent.get_action(state)

        # 환경에서 행동 실행
        next_state, reward, terminated, truncated, info = env_video.step(action)
        done = terminated | truncated

        # 결과 기록
        frames.append(env_video.render())
        total_reward += reward
        next_state = preprocess(next_state)
        state = next_state
        steps += 1

    # 마지막 프레임 30개 더 추가 (자연스러운 종료)
    for _ in range(30):
        frames.append(env_video.render())

    # 비디오 저장
    try:
        imageio.mimsave(filename, frames, fps=30)
        print(f"게임 영상이 {filename}에 저장되었습니다. Steps: {steps}, Total reward: {total_reward:.2f}")
    except Exception as e:
        print(f"비디오 저장 실패: {e}")
        # GIF로 대체 시도
        try:
            gif_filename = filename.replace('.mp4', '.gif')
            imageio.mimsave(gif_filename, frames, fps=30)
            print(f"GIF로 저장되었습니다: {gif_filename}")
        except Exception as e2:
            print(f"GIF 저장도 실패: {e2}")

    env_video.close()
    return total_reward, steps


# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 학습 시작 시간 기록
start_time = time.time()
print("학습을 시작합니다...")

episodes = 1000
sync_interval = 10

env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
agent = DQNAgent(device=device)
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    state = preprocess(state)
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess(next_state)
        done = terminated | truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode : {}, total reward : {}".format(episode, total_reward))

# 학습 완료 시간 기록 및 출력
end_time = time.time()
training_time = end_time - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)
print(f"학습 완료! 총 학습 시간: {hours}시간 {minutes}분 {seconds}초")

# 100회 이동평균 계산
moving_avg_100 = calculate_moving_average(reward_history, 100)

# 그래프 저장 (에피소드별 보상 + 100회 평균)
plt.figure(figsize=(12, 8))

# 에피소드별 보상 그래프
plt.subplot(2, 1, 1)
plt.plot(range(len(reward_history)), reward_history, alpha=0.7, color='blue', linewidth=0.8)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')
plt.grid(True)

# 100회 이동평균 그래프
plt.subplot(2, 1, 2)
plt.plot(range(len(moving_avg_100)), moving_avg_100, color='red', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('100-Episode Moving Average')
plt.title('100-Episode Moving Average Rewards')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 결합된 그래프도 저장
plt.figure(figsize=(10, 6))
plt.plot(range(len(reward_history)), reward_history, alpha=0.5, color='blue', linewidth=0.8, label='Episode Reward')
plt.plot(range(len(moving_avg_100)), moving_avg_100, color='red', linewidth=2, label='100-Episode Moving Average')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('CarRacing DQN Training Results')
plt.legend()
plt.grid(True)
plt.savefig('combined_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("그래프가 저장되었습니다: training_results.png, combined_training_results.png")

# 게임 영상 저장
print("게임 플레이 영상을 생성 중...")
save_gameplay_video(agent, "racing_gameplay.mp4")

# 원래 코드의 마지막 부분 (사람이 보는 화면)
env2 = gym.make('CarRacing-v2', continuous=False, render_mode='human')

agent.epsilon = 0
state = env.reset()[0]
state = preprocess(state)

done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated | truncated

    next_state = preprocess(next_state)
    state = next_state
    total_reward += reward
    env2.render()
print('Total Reward : ', total_reward)