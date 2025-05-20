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

        # 탐색을 위한 파라미터
        self.epsilon = 0.2  # 초기 탐험 확률
        self.epsilon_decay = 0.995  # 탐험 확률 감소율
        self.epsilon_min = 0.01  # 최소 탐험 확률

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

    def get_action(self, state, training=True):
        """현재 정책에 따라 행동 선택"""
        state = self.preprocess_state(state)
        state = state[np.newaxis, :]

        # 훈련 중 epsilon-greedy 전략 적용
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size), None

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

        # Actor (정책) 업데이트 - action_prob가 None이 아닌 경우만
        if action_prob is not None:
            loss_pi = -F.log(action_prob) * td_error
            self.pi.cleargrads()
            loss_pi.backward()
            self.optimizer_pi.update()

        # 탐험 확률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return td_error


# Mountain Car 환경에 대한 사용자 정의 환경 래퍼
class MountainCarWithCustomResetWrapper(gym.Wrapper):
    def __init__(self, env, advanced_start_prob=0.5, min_position=-0.6, min_velocity=0.0):
        super().__init__(env)
        self.advanced_start_prob = advanced_start_prob  # 유리한 시작 위치를 사용할 확률
        self.min_position = min_position  # 유리한 시작 위치의 최소값
        self.min_velocity = min_velocity  # 유리한 시작 속도의 최소값
        self.total_resets = 0
        self.advanced_resets = 0

    def reset(self, **kwargs):
        self.total_resets += 1
        state, info = self.env.reset(**kwargs)

        # advanced_start_prob 확률로 유리한 시작 위치 사용
        if np.random.random() < self.advanced_start_prob:
            self.advanced_resets += 1
            # 왼쪽 언덕 높은 위치에서 시작, 오른쪽 방향으로 약간의 속도
            position = np.random.uniform(self.min_position, -0.4)  # -0.6 ~ -0.4 사이
            velocity = np.random.uniform(self.min_velocity, 0.04)  # 0.0 ~ 0.04 사이 (오른쪽 방향)
            self.env.unwrapped.state = np.array([position, velocity])
            state = self.env.unwrapped.state

        if self.total_resets % 100 == 0:
            print(f"Advanced resets: {self.advanced_resets}/{self.total_resets} "
                  f"({self.advanced_resets / self.total_resets * 100:.1f}%)")

        return state, info


# 보상 쉐이핑 함수
def shaped_reward(state, next_state, reward, done):
    """Mountain Car 문제에 강화된 보상 쉐이핑"""
    position, velocity = next_state
    old_position, old_velocity = state

    # 기본 보상: 매 스텝마다 -1
    base_reward = reward

    # === 높이 기반 보상 ===
    # 높은 위치에 더 큰 보상 (위치 범위: -1.2 ~ 0.6)
    # 정규화된 위치값으로 변환: 0 ~ 1.8
    normalized_position = position + 1.2
    # 높이 보상: 최대 3점
    height_reward = normalized_position * 1.5

    # === 속도 기반 보상 ===
    # 속도의 절대값에 비례한 보상 (속도 범위: -0.07 ~ 0.07)
    # 절대 속도가 클수록 더 많은 보상
    speed_reward = abs(velocity) * 30  # 최대 약 2점

    # === 방향 기반 보상 ===
    direction_reward = 0

    # 왼쪽 끝으로 갈 때 보상 (진자 운동 유도)
    if position < -1.0 and velocity < 0:
        direction_reward += 2.0

    # 중간 이상에서 오른쪽으로 갈 때 보상
    if position > -0.5 and velocity > 0:
        direction_reward += 3.0

    # === 운동 에너지 변화 보상 ===
    # 속도가 증가할 때 보상
    energy_change = abs(velocity) - abs(old_velocity)
    if energy_change > 0:
        energy_reward = energy_change * 20  # 속도 증가에 큰 보상
    else:
        energy_reward = 0

    # === 특별 지점 보상 ===
    position_milestone_reward = 0

    # 왼쪽 끝에 도달하면 큰 보상
    if position <= -1.1:
        position_milestone_reward += 3.0

    # 오른쪽으로 높이 올라갈수록 더 큰 보상
    if position >= 0.0:
        position_milestone_reward += 2.0
    if position >= 0.3:
        position_milestone_reward += 3.0
    if position >= 0.4:
        position_milestone_reward += 5.0

    # === 목표 달성 보상 ===
    goal_reward = 0
    if done and position >= 0.5:
        goal_reward = 50.0  # 목표 달성 시 매우 큰 보상

    # 모든 보상 합산
    total_shaped_reward = (
            base_reward +  # 기본 보상 (-1)
            height_reward +  # 높이 보상 (최대 3)
            speed_reward +  # 속도 보상 (최대 2)
            direction_reward +  # 방향 보상 (최대 5)
            energy_reward +  # 에너지 변화 보상 (동적)
            position_milestone_reward +  # 위치 이정표 보상 (최대 13)
            goal_reward  # 목표 보상 (50)
    )

    # 디버깅용 출력 (학습 초기에만 사용, 이후 주석 처리)
    if np.random.random() < 0.001:  # 0.1% 확률로 출력
        print(f"Reward breakdown - Base: {base_reward:.1f}, Height: {height_reward:.1f}, "
              f"Speed: {speed_reward:.1f}, Direction: {direction_reward:.1f}, "
              f"Energy: {energy_reward:.1f}, Milestone: {position_milestone_reward:.1f}, "
              f"Goal: {goal_reward:.1f}, Total: {total_shaped_reward:.1f}")

    return total_shaped_reward


# 비디오 생성 함수
def record_video(agent, env, filename="mountain_car_ac.mp4"):
    """학습된 에이전트의 성능을 비디오로 기록"""
    # 비디오 기록용 환경 생성 (원래 환경과 동일한 설정)
    video_env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state, _ = video_env.reset()
    done = False
    frames = []
    total_reward = 0
    steps = 0

    while not done and steps < 200:  # 최대 200 스텝
        # 행동 선택 (학습된 정책 사용)
        action, _ = agent.get_action(state, training=False)

        # 환경에서 행동 실행
        next_state, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated

        # 결과 기록
        frames.append(video_env.render())
        total_reward += reward
        state = next_state
        steps += 1

    # 마지막 프레임 몇 개 더 추가
    for _ in range(30):
        frames.append(video_env.render())

    # 비디오 저장
    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved as {filename}, Steps: {steps}, Total reward: {total_reward}")
    return total_reward, steps


# 학습 함수
def train(env_name="MountainCar-v0", gamma=0.99, lr_pi=0.0002, lr_v=0.001,
          num_episodes=600, use_reward_shaping=True, use_custom_reset=True):
    """Actor-Critic 에이전트 학습"""
    # 기본 환경 생성
    env = gym.make(env_name)

    # 사용자 정의 초기화 래퍼 적용 (선택적)
    if use_custom_reset:
        env = MountainCarWithCustomResetWrapper(env, advanced_start_prob=0.7)

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
    solved_episode = None

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        # 에피소드 실행
        while not done and steps < 200:  # MountainCar-v0는 최대 200 스텝
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
            print(f"Episode {episode}: Steps = {steps}, Reward = {total_reward:.2f}, "
                  f"Avg(10) = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

        # 100 에피소드마다 비디오 기록
        if episode % 100 == 0 and episode > 0:
            record_video(agent, env, f"mountain_car_ac_ep{episode}.mp4")

        # 학습 종료 조건
        if len(rewards_history) >= 100 and np.mean(rewards_history[-100:]) > -110 and solved_episode is None:
            solved_episode = episode
            print(f"\n********** 문제 해결! Episode {episode} **********")
            print(f"100 에피소드 평균 보상: {np.mean(rewards_history[-100:]):.2f}")
            # 문제 해결 시 비디오 기록
            record_video(agent, env, f"mountain_car_ac_solved.mp4")

    # 학습 결과 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    if solved_episode is not None:
        plt.axvline(x=solved_episode, color='r', linestyle='--', label=f'Solved at episode {solved_episode}')
        plt.legend()
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
    record_video(agent, env, f"mountain_car_ac_final.mp4")

    # 최종 성능 평가
    eval_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        eval_rewards.append(total_reward)

    avg_eval_reward = np.mean(eval_rewards)
    print(f"\nFinal evaluation over 10 episodes: {avg_eval_reward:.2f}")

    return agent, rewards_history, steps_history


# 실험 함수
def run_experiment(custom_reset=True, reward_shaping=True, gamma=0.99, lr_pi=0.0002, lr_v=0.001):
    """학습 설정을 변경하며 실험 실행"""
    print(f"=== Running experiment with custom_reset={custom_reset}, reward_shaping={reward_shaping}, "
          f"gamma={gamma}, lr_pi={lr_pi}, lr_v={lr_v} ===")

    agent, rewards, steps = train(
        gamma=gamma,
        lr_pi=lr_pi,
        lr_v=lr_v,
        use_custom_reset=custom_reset,
        use_reward_shaping=reward_shaping
    )

    # 평균 및 최고 보상 기록
    if len(rewards) >= 100:
        last_100_avg = np.mean(rewards[-100:])
    else:
        last_100_avg = np.mean(rewards)

    best_reward = np.max(rewards)

    print(f"Results: Last 100 Avg = {last_100_avg:.2f}, Best = {best_reward:.2f}")
    return agent, rewards, steps


# 메인 함수
if __name__ == "__main__":
    # 사용자 정의 초기화 + 보상 쉐이핑 (가장 효과적)
    run_experiment(custom_reset=True, reward_shaping=True)

    # 나머지 실험 조합은 주석 처리 (원하는 경우 주석 해제 가능)
    # run_experiment(custom_reset=True, reward_shaping=False)  # 초기화만 사용
    # run_experiment(custom_reset=False, reward_shaping=True)  # 보상 쉐이핑만 사용
    # run_experiment(custom_reset=False, reward_shaping=False)  # 기본 설정