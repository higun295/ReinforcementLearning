import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import gym
import matplotlib.pyplot as plt
import imageio
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


# 이산 행동 정책 네트워크 (Actor)
class PolicyNet(Model):
    def __init__(self, action_size=3):  # MountainCar-v0는 3개의 이산 행동
        super().__init__()
        self.l1 = L.Linear(64)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)  # 소프트맥스로 행동 확률 출력
        return x


# 가치 함수 네트워크 (Critic)
class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(64)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


# Actor-Critic 에이전트
class ActorCriticAgent:
    def __init__(self, state_size=2, action_size=3, gamma=0.99, lr_pi=0.0002, lr_v=0.001):
        self.gamma = gamma
        self.action_size = action_size

        # 네트워크 초기화
        self.pi = PolicyNet(action_size)
        self.v = ValueNet()

        # 옵티마이저 설정
        self.optimizer_pi = optimizers.Adam(lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(lr_v).setup(self.v)

        # 상태 정규화를 위한 변수
        self.state_mean = None
        self.state_std = None

    def preprocess_state(self, state):
        """상태 정규화"""
        if self.state_mean is not None and self.state_std is not None:
            return (state - self.state_mean) / (self.state_std + 1e-5)
        return state

    def compute_state_stats(self, env, num_samples=10000):
        """상태 공간 통계 계산"""
        states = []
        for _ in range(num_samples):
            states.append(env.observation_space.sample())
        states = np.array(states)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0)
        print(f"State mean: {self.state_mean}, State std: {self.state_std}")

    def get_action(self, state):
        """현재 정책에 따라 행동 선택"""
        state = self.preprocess_state(state)
        state = state[np.newaxis, :]
        action_probs = self.pi(state)
        action_probs = action_probs[0]

        # 확률에 따라 행동 샘플링
        action = np.random.choice(self.action_size, p=action_probs.data)
        return action, action_probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        """에이전트 업데이트"""
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)

        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # 다음 상태의 가치 계산
        next_value = self.v(next_state)

        # TD 타겟 계산 - 배열 형태 유지
        if done:
            # 종료 상태면 보상만
            target = np.array([[reward]])
        else:
            # 보상 + 감마 * 다음 상태 가치
            target = np.array([[reward + self.gamma * next_value.data[0][0]]])

        # 현재 상태의 가치 계산
        current_value = self.v(state)

        # TD 오류 계산 (데이터만 사용)
        td_error = target[0][0] - current_value.data[0][0]

        # Critic (가치 함수) 업데이트 - target을 배열로 변환
        loss_v = F.mean_squared_error(current_value, target)
        self.v.cleargrads()
        loss_v.backward()
        self.optimizer_v.update()

        # Actor (정책) 업데이트 - td_error는 스칼라이므로 그대로 사용
        loss_pi = -F.log(action_prob) * td_error
        self.pi.cleargrads()
        loss_pi.backward()
        self.optimizer_pi.update()

        return td_error


# 보상 쉐이핑 함수
def shaped_reward(state, next_state, reward, done):
    """Mountain Car 문제에 맞는 보상 쉐이핑"""
    position, velocity = next_state

    # 높이에 따른 보상
    height_reward = position * 0.5

    # 속도와 위치를 고려한 보상
    direction_reward = position * velocity * 0.1

    # 기본 보상 + 쉐이핑 보상
    shaped_reward = reward + height_reward + direction_reward

    # 목표 도달 시 큰 보상
    if done and position >= 0.5:
        shaped_reward += 10.0

    return shaped_reward


# 비디오 생성 함수
def record_video(agent, env_name="MountainCar-v0", filename="mountain_car_ac.mp4"):
    """학습된 에이전트의 성능을 비디오로 기록"""
    env = gym.make(env_name, render_mode="rgb_array")
    state = env.reset()[0]
    done = False
    frames = []
    total_reward = 0

    while not done:
        # 행동 선택 (학습된 정책 사용)
        action, _ = agent.get_action(state)

        # 환경에서 행동 실행
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 결과 기록
        frames.append(env.render())
        total_reward += reward
        state = next_state

    # 마지막 프레임 몇 개 더 추가
    for _ in range(30):
        frames.append(env.render())

    # 비디오 저장
    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved as {filename}, Total reward: {total_reward}")
    return total_reward


# 학습 함수
def train(env_name="MountainCar-v0", gamma=0.99, lr_pi=0.0002, lr_v=0.001,
          num_episodes=1000, max_steps=1000, use_reward_shaping=True):
    """Actor-Critic 에이전트 학습"""
    env = gym.make(env_name)
    agent = ActorCriticAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        gamma=gamma,
        lr_pi=lr_pi,
        lr_v=lr_v
    )

    # 상태 정규화를 위한 통계 계산
    agent.compute_state_stats(env)

    # 학습 기록
    rewards_history = []
    steps_history = []
    best_reward = -float('inf')

    for episode in range(num_episodes):
        state = env.reset()[0]
        total_reward = 0
        steps = 0
        done = False

        # 에피소드 실행
        while not done and steps < max_steps:
            # 행동 선택 및 실행
            action, action_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # 보상 쉐이핑 적용 (옵션)
            if use_reward_shaping:
                modified_reward = shaped_reward(state, next_state, reward, done)
            else:
                modified_reward = reward

            # 에이전트 업데이트
            td_error = agent.update(state, action_prob, modified_reward, next_state, done)

            # 상태 업데이트 및 보상 누적
            state = next_state
            total_reward += reward  # 원래 보상으로 성능 측정

        # 에피소드 결과 기록
        rewards_history.append(total_reward)
        steps_history.append(steps)

        # 주기적으로 학습 상태 출력
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
            print(f"Episode {episode}: Steps = {steps}, Reward = {total_reward:.2f}, Avg(10) = {avg_reward:.2f}")

        # 최고 성능 기록
        if total_reward > best_reward and total_reward > -150:
            best_reward = total_reward
            # 최고 성능일 때 비디오 기록 (선택적)
            if episode > 100 and episode % 100 == 0:
                record_video(agent, env_name, f"mountain_car_ac_ep{episode}.mp4")

        # 학습 종료 조건
        if len(rewards_history) >= 100 and np.mean(rewards_history[-100:]) > -110:
            print(f"\n********** 문제 해결! Episode {episode} **********")
            print(f"100 에피소드 평균 보상: {np.mean(rewards_history[-100:]):.2f}")
            break

    # 학습 결과 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Rewards (gamma={gamma}, lr_pi={lr_pi}, lr_v={lr_v})')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps_history)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'mountain_car_ac_g{gamma}_lrpi{lr_pi}_lrv{lr_v}.png')
    plt.show()

    # 최종 성능 비디오 기록
    record_video(agent, env_name, f"mountain_car_ac_final.mp4")

    return agent, rewards_history, steps_history


# 단일 하이퍼파라미터 실험 함수
def experiment(gamma=0.99, lr_pi=0.0002, lr_v=0.001, episodes=600):
    """단일 하이퍼파라미터 세트 테스트"""
    print(f"=== Running experiment with gamma={gamma}, lr_pi={lr_pi}, lr_v={lr_v} ===")
    agent, rewards, steps = train(
        gamma=gamma,
        lr_pi=lr_pi,
        lr_v=lr_v,
        num_episodes=episodes
    )

    # 평균 및 최고 보상 기록
    avg_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
    best_reward = np.max(rewards)

    print(f"Results: Avg last 50 = {avg_reward:.2f}, Best = {best_reward:.2f}")
    return avg_reward, best_reward


# 메인 함수
if __name__ == "__main__":
    # 단일 실험 실행
    experiment(gamma=0.99, lr_pi=0.0002, lr_v=0.001, episodes=600)

    # 더 많은 실험을 원한다면 다음과 같이 여러 조합 테스트 가능

    # 감마 변화에 따른 영향 테스트
    # experiment(gamma=0.999, lr_pi=0.0002, lr_v=0.001, episodes=600)

    # 학습률 변화에 따른 영향 테스트
    # experiment(gamma=0.99, lr_pi=0.0005, lr_v=0.001, episodes=600)
    # experiment(gamma=0.99, lr_pi=0.0002, lr_v=0.002, episodes=600)