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
        # 더 큰 네트워크 사용
        self.l1 = L.Linear(256)
        self.l2 = L.Linear(256)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# Kaggle 노트북에서 영감을 받은 보상 함수
def compute_reward(state, next_state, done):
    # 기본 보상
    reward = -1.0

    # 목표 도달 시 큰 보상
    if done and next_state[0] >= 0.5:
        reward += 500.0

    # 진자 운동 장려 (Kaggle 노트북처럼)
    # 오른쪽으로 가는 중인데 속도가 양수이면 (가속 중)
    if next_state[1] > 0 and next_state[0] > state[0]:
        reward += 10.0 * abs(next_state[1])  # 속도에 비례

    # 왼쪽으로 가는 중인데 속도가 음수이면 (산 반대편으로 이동 중)
    elif next_state[1] < 0 and next_state[0] < -0.5:
        reward += 10.0 * abs(next_state[1])  # 속도에 비례

    # 높이에 대한 보상 (더 높이 올라갈수록 더 많은 보상)
    position_reward = (next_state[0] + 1.2) / 1.8  # -1.2~0.6 범위를 0~1로 정규화
    reward += position_reward * 10.0

    return reward


class DQNAgent:
    def __init__(self):
        # Kaggle 노트북에서 영감을 받은 하이퍼파라미터
        self.gamma = 0.99  # 감가율
        self.learning_rate = 0.0005  # 학습률
        self.epsilon_start = 1.0  # 초기 입실론
        self.epsilon_min = 0.01  # 최소 입실론
        self.epsilon_decay = 0.995  # 입실론 감소율
        self.epsilon = self.epsilon_start
        self.buffer_size = 100000  # 버퍼 크기
        self.batch_size = 64  # 배치 크기
        self.action_size = 3  # 행동 크기 (왼쪽, 정지, 오른쪽)

        # 네트워크 및 버퍼 초기화
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.learning_rate)
        self.optimizer.setup(self.qnet)

        # 타깃 네트워크 초기화
        self.sync_qnet()

        # 학습 관련 변수
        self.update_count = 0
        self.goals_reached = 0  # 목표 도달 횟수 추적

    def get_action(self, state):
        # 입실론-그리디 정책
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        # 경험 추가
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        # 미니배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        # 현재 Q 값
        qs = self.qnet(states)
        q = qs[np.arange(self.batch_size), actions]

        # Double DQN 구현
        # 현재 네트워크에서 다음 행동 선택
        next_qs = self.qnet(next_states)
        next_actions = next_qs.data.argmax(axis=1)

        # 타깃 네트워크에서 다음 행동의 Q 값 평가
        next_qs_target = self.qnet_target(next_states)
        next_q = next_qs_target[np.arange(self.batch_size), next_actions]
        next_q.unchain()

        # TD 타깃 계산 (Kaggle 노트북에서 언급된 방식)
        target = rewards + (1 - dones) * self.gamma * next_q

        # 네트워크 업데이트
        loss = F.mean_squared_error(q, target)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        # 입실론 감소
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 업데이트 카운트 증가
        self.update_count += 1

        return loss

    def sync_qnet(self):
        # 타깃 네트워크 동기화
        self.qnet_target = copy.deepcopy(self.qnet)

    def save_model(self, filename):
        # 모델 저장 함수 (dezero에서는 구현 필요)
        np.save(f'{filename}_weights', [p.data for p in self.qnet.params()])
        print(f"모델 저장 완료: {filename}")


# 학습 진행 및 결과 시각화
def main():
    # 환경 및 에이전트 초기화
    env = gym.make('MountainCar-v0')
    agent = DQNAgent()

    # 학습 파라미터
    episodes = 500  # Kaggle 노트북에서는 400 에피소드 사용
    sync_interval = 5  # 타깃 네트워크 동기화 주기
    reward_history = []  # 보상 이력
    steps_history = []  # 스텝 이력
    goal_history = []  # 목표 도달 이력

    # 랜덤 경험 수집으로 버퍼 초기화
    print("=== 경험 버퍼 초기화 중 ===")
    state = env.reset()[0]
    for _ in range(1000):
        action = np.random.randint(0, agent.action_size)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        modified_reward = compute_reward(state, next_state, done)
        agent.replay_buffer.add(state, action, modified_reward, next_state, done)
        if done:
            state = env.reset()[0]
        else:
            state = next_state

    # 학습 시작
    print("=== 학습 시작 ===")
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        original_reward = 0
        step_count = 0
        done = False
        goal_reached = False

        while not done and step_count < 200:  # 최대 200 스텝 (Mountain Car 기본값)
            # 행동 선택 및 실행
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 목표 도달 확인
            if next_state[0] >= 0.5:
                goal_reached = True
                agent.goals_reached += 1

            # 수정된 보상 계산
            modified_reward = compute_reward(state, next_state, done)

            # 에이전트 업데이트
            agent.update(state, action, modified_reward, next_state, done)

            # 상태 및 보상 업데이트
            state = next_state
            total_reward += modified_reward
            original_reward += reward
            step_count += 1

            # 주기적으로 학습 결과 시각화 (옵션)
            if episode % 50 == 0 and episode > 0:
                env.render()

        # 타깃 네트워크 동기화
        if episode % sync_interval == 0:
            agent.sync_qnet()

        # 결과 기록
        reward_history.append(original_reward)
        steps_history.append(step_count)
        goal_history.append(1 if goal_reached else 0)

        # 진행 상황 출력
        if episode % 10 == 0 or goal_reached:
            recent_goals = sum(goal_history[-100:]) if len(goal_history) >= 100 else sum(goal_history)
            avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
            print(f"에피소드: {episode}, 총 보상: {original_reward:.2f}, 수정된 보상: {total_reward:.2f}, "
                  f"스텝: {step_count}, 입실론: {agent.epsilon:.4f}, "
                  f"목표 도달: {'예' if goal_reached else '아니오'}, 최근 목표 도달 횟수: {recent_goals}")

        # 모델 저장 (주기적으로)
        if episode % 100 == 99:
            agent.save_model(f"mountain_car_ep{episode + 1}")

    # 학습 종료
    env.close()

    # 결과 시각화
    plt.figure(figsize=(15, 10))

    # 1. 총 보상 그래프
    plt.subplot(2, 2, 1)
    plt.plot(reward_history)
    plt.xlabel('에피소드')
    plt.ylabel('총 보상')
    plt.title('에피소드별 총 보상')
    plt.grid(True)

    # 2. 이동 평균 보상 그래프
    plt.subplot(2, 2, 2)
    moving_avg = []
    window_size = 20
    for i in range(len(reward_history) - window_size + 1):
        moving_avg.append(np.mean(reward_history[i:i + window_size]))
    plt.plot(range(window_size - 1, len(reward_history)), moving_avg)
    plt.xlabel('에피소드')
    plt.ylabel(f'{window_size}개 에피소드 평균 보상')
    plt.title('이동 평균 보상')
    plt.grid(True)

    # 3. 스텝 수 그래프
    plt.subplot(2, 2, 3)
    plt.plot(steps_history)
    plt.xlabel('에피소드')
    plt.ylabel('스텝 수')
    plt.title('에피소드별 스텝 수')
    plt.grid(True)

    # 4. 목표 도달 여부 그래프
    plt.subplot(2, 2, 4)
    plt.plot(goal_history)
    plt.xlabel('에피소드')
    plt.ylabel('목표 도달 (1=도달, 0=실패)')
    plt.title('에피소드별 목표 도달 여부')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('mountain_car_training_results.png')
    plt.show()

    # 학습된 에이전트 테스트
    test_agent(agent)


def test_agent(agent):
    print("\n=== 학습된 에이전트 테스트 ===")
    env = gym.make('MountainCar-v0', render_mode='human')

    # 탐색 없이 테스트
    agent.epsilon = 0

    test_results = []

    for test_ep in range(5):  # 5번 테스트
        state = env.reset()[0]
        total_reward = 0
        step_count = 0
        done = False
        goal_reached = False

        while not done and step_count < 200:
            # 행동 선택
            action = agent.get_action(state)

            # 환경 진행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 목표 도달 확인
            if next_state[0] >= 0.5:
                goal_reached = True

            # 상태 및 보상 업데이트
            state = next_state
            total_reward += reward
            step_count += 1

            # 렌더링
            env.render()

        result = {
            'episode': test_ep + 1,
            'reward': total_reward,
            'steps': step_count,
            'goal_reached': goal_reached
        }
        test_results.append(result)

        print(
            f"테스트 {test_ep + 1}/5, 총 보상: {total_reward:.2f}, 스텝: {step_count}, 목표 도달: {'예' if goal_reached else '아니오'}")

    env.close()

    # 테스트 결과 요약
    success_rate = sum([1 for r in test_results if r['goal_reached']]) / len(test_results) * 100
    avg_steps = np.mean([r['steps'] for r in test_results])
    avg_reward = np.mean([r['reward'] for r in test_results])

    print(f"\n테스트 결과 요약:")
    print(f"성공률: {success_rate:.1f}%")
    print(f"평균 스텝: {avg_steps:.1f}")
    print(f"평균 보상: {avg_reward:.2f}")

    print("테스트 완료!")


if __name__ == "__main__":
    main()