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

# 안정성을 위한 작은 상수
EPS = 1e-8


class PolicyNet(Model):
    def __init__(self, state_size=2, action_size=3):
        super().__init__()
        self.l1 = L.Linear(32)  # 더 작은 네트워크
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        # x = F.softmax(x)  # 기존 softmax
        x = F.softmax(F.clip(x, -10, 10))  # 클리핑 적용 softmax
        return x


class ValueNet(Model):
    def __init__(self, state_size=2):
        super().__init__()
        self.l1 = L.Linear(32)  # 더 작은 네트워크
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self, state_size=2, action_size=3, gamma=0.99, lr_pi=0.0001, lr_v=0.0005):
        self.gamma = gamma
        self.lr_pi = lr_pi
        self.lr_v = lr_v
        self.state_size = state_size
        self.action_size = action_size

        self.pi = PolicyNet(state_size, action_size)
        self.v = ValueNet(state_size)
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

        # 탐색을 위한 파라미터
        self.epsilon = 0.2  # 높은 epsilon 값으로 시작

    def get_action(self, state, training=True):
        state = state[np.newaxis, :]

        # epsilon-greedy 탐색 추가
        if training and np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
            # 정책의 확률 계산은 여전히 필요
            probs = self.pi(state)
            probs = probs[0]
            return action, probs[action]

        probs = self.pi(state)
        probs = probs[0]

        # NaN 확인 및 안전하게 처리
        if np.isnan(probs.data).any() or np.sum(probs.data) == 0:
            print("Warning: NaN in probabilities - using uniform distribution")
            # NaN이 있으면 균등 분포로 대체
            uniform_probs = np.ones(self.action_size) / self.action_size
            action = np.random.choice(self.action_size, p=uniform_probs)
            return action, self.pi(state)[0][action]

        # 확률이 제대로 합이 1이 되는지 확인
        probs_sum = np.sum(probs.data)
        if abs(probs_sum - 1.0) > 0.01:  # 합이 1에서 많이 벗어나는 경우
            print(f"Warning: Probability sum is {probs_sum}, normalizing")
            # 명시적으로 정규화
            normalized_probs = probs.data / probs_sum
            action = np.random.choice(self.action_size, p=normalized_probs)
            return action, probs[action]

        action = np.random.choice(self.action_size, p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # TD 목표 계산
        next_value = self.v(next_state)
        if np.isnan(next_value.data).any():
            print("Warning: NaN in next_value - skipping update")
            return 0  # NaN이 발생하면 업데이트 건너뛰기

        target = reward + self.gamma * next_value * (1 - done)
        target.unchain()
        v = self.v(state)

        # 시간적 차이 오차(TD error)
        td_error = target - v

        # Critic (Value Network) 업데이트
        loss_v = F.mean_squared_error(v, target)

        # Actor (Policy Network) 업데이트
        # log 함수에 작은 값을 더해 0으로 나누기 방지
        safe_log_prob = F.log(action_prob + EPS)
        loss_pi = -safe_log_prob * F.clip(td_error.data, -1.0, 1.0)  # TD 오류 클리핑

        # 그래디언트 초기화 및 역전파
        self.v.cleargrads()
        loss_v.backward()
        self.optimizer_v.update()

        self.pi.cleargrads()
        loss_pi.backward()
        self.optimizer_pi.update()

        return td_error.data if not np.isnan(td_error.data) else 0


# 보상 쉐이핑을 단순화해서 수치적 안정성 향상
def shaped_reward(state, next_state, reward, done):
    """
    Mountain Car 문제에 맞는 단순화된 보상 쉐이핑 함수
    """
    position, velocity = next_state

    # 높이에 따른 보상 (높을수록 좋음)
    height_reward = position * 0.1  # 단순화된 높이 보상

    # 기본 보상 + 쉐이핑 보상
    shaped_reward = reward + height_reward

    # 목표 도달 시 보상
    if done and position >= 0.5:  # 목표 지점에 도달
        shaped_reward += 10.0

    return shaped_reward


def train_and_evaluate(gamma=0.99, lr_pi=0.0001, lr_v=0.0005, episodes=300, max_steps=200):
    env = gym.make("MountainCar-v0")
    agent = Agent(gamma=gamma, lr_pi=lr_pi, lr_v=lr_v)
    reward_history = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step = 0

        # NaN 감지를 위한 플래그
        nan_detected = False

        while not done and step < max_steps:
            try:
                action, prob = agent.get_action(state)

                # NaN 체크
                if np.isnan(prob.data):
                    print(f"NaN detected in prob at episode {episode}, step {step}")
                    nan_detected = True
                    break

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # 보상 쉐이핑 적용
                modified_reward = shaped_reward(state, next_state, reward, done)

                # 에이전트 업데이트
                td_error = agent.update(state, prob, modified_reward, next_state, done)

                # NaN 체크
                if np.isnan(td_error).any():
                    print(f"NaN detected in td_error at episode {episode}, step {step}")
                    nan_detected = True
                    break

                state = next_state
                total_reward += reward  # 원래 보상을 기록
                step += 1

            except Exception as e:
                print(f"Error at episode {episode}, step {step}: {str(e)}")
                nan_detected = True
                break

        # NaN이 감지되면 모델 재초기화 고려
        if nan_detected:
            print(f"Reinitializing agent at episode {episode}")
            agent = Agent(gamma=gamma, lr_pi=lr_pi, lr_v=lr_v)

        # 에피소드가 끝날 때마다 epsilon 감소 (탐색 감소)
        if episode > 50:  # 처음 50 에피소드는 충분히 탐색
            agent.epsilon = max(0.05, agent.epsilon * 0.995)  # 최소 0.05까지 감소

        reward_history.append(total_reward)
        if episode % 10 == 0:  # 더 자주 로그 출력
            avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
            print(
                f"Episode {episode}: Reward = {total_reward:.1f}, Avg(10) = {avg_reward:.1f}, Epsilon = {agent.epsilon:.3f}")

    # 학습 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Mountain Car (Actor-Critic): gamma={gamma}, lr_pi={lr_pi}, lr_v={lr_v}")
    plt.grid(True)
    plt.savefig(f"mountain_car_ac_g{gamma}_lrpi{lr_pi}_lrv{lr_v}.png")
    plt.show()

    return agent, reward_history


def test_agent(agent, env_name="MountainCar-v0", render=False, num_episodes=3):
    """학습된 에이전트 테스트"""
    env = gym.make(env_name, render_mode="human" if render else None)
    test_rewards = []

    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step = 0

        while not done and step < 200:  # 최대 200 스텝으로 제한
            action, _ = agent.get_action(state, training=False)  # 탐색 없이 정책 따름
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step += 1

            if render:
                env.render()

        test_rewards.append(total_reward)

    avg_reward = np.mean(test_rewards)
    return avg_reward


def render_and_save_video(agent, filename="mountaincar_actor_critic.mp4"):
    """학습된 에이전트로 비디오 생성"""
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state = env.reset()[0]
    done = False
    frames = []
    total_reward = 0
    step = 0

    while not done and step < 200:  # 최대 200 스텝 제한
        action, _ = agent.get_action(state, training=False)  # 탐색 없이
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        frames.append(env.render())
        step += 1

    # 마지막 장면을 30프레임 더 추가
    for _ in range(30):
        frames.append(env.render())

    env.close()
    print(f"Video Total Reward: {total_reward:.1f}, Steps: {step}")
    imageio.mimsave(filename, frames, fps=30)

    return total_reward, step


def hyperparameter_search():
    """하이퍼파라미터 탐색 함수"""
    # 축소된 하이퍼파라미터 탐색 공간률
    gammas = [0.99, 0.999]  # 2개 감마
    lr_pis = [0.0001, 0.0005]  # 2개 액터 학습률
    lr_vs = [0.0005, 0.001]  # 2개 크리틱 학습

    best_reward = -float('inf')
    best_params = {}
    best_agent = None
    results = []

    # 각 조합에 대해 테스트
    for gamma in gammas:
        for lr_pi in lr_pis:
            for lr_v in lr_vs:
                print(f"\nTesting with gamma={gamma}, lr_pi={lr_pi}, lr_v={lr_v}")

                # 학습 실행 (더 적은 에피소드로 빠르게 테스트)
                agent, rewards = train_and_evaluate(
                    gamma=gamma,
                    lr_pi=lr_pi,
                    lr_v=lr_v,
                    episodes=200  # 탐색 단계에서는 적은 에피소드로 빠르게 테스트
                )

                # 테스트 성능 측정
                test_reward = test_agent(agent, render=False, num_episodes=3)
                print(f"Test average reward: {test_reward:.2f}")

                # 결과 저장
                results.append({
                    'gamma': gamma,
                    'lr_pi': lr_pi,
                    'lr_v': lr_v,
                    'avg_reward': test_reward
                })

                # 최고 성능 저장
                if test_reward > best_reward:
                    best_reward = test_reward
                    best_params = {'gamma': gamma, 'lr_pi': lr_pi, 'lr_v': lr_v}
                    best_agent = agent

    # 결과 정렬 및 출력
    results.sort(key=lambda x: x['avg_reward'], reverse=True)
    print("\nAll parameter combinations (sorted by performance):")
    for i, result in enumerate(results):
        print(
            f"{i + 1}. gamma={result['gamma']}, lr_pi={result['lr_pi']}, lr_v={result['lr_v']}: {result['avg_reward']:.2f}")

    print("\nBest hyperparameters:")
    print(f"gamma={best_params['gamma']}, lr_pi={best_params['lr_pi']}, lr_v={best_params['lr_v']}")
    print(f"Best test reward: {best_reward:.2f}")

    return best_params, best_agent, results


def final_training(params, episodes=500):
    """최적 하이퍼파라미터로 최종 학습"""
    print(
        f"\nFinal training with best parameters: gamma={params['gamma']}, lr_pi={params['lr_pi']}, lr_v={params['lr_v']}")

    # 최종 학습 실행
    agent, rewards = train_and_evaluate(
        gamma=params['gamma'],
        lr_pi=params['lr_pi'],
        lr_v=params['lr_v'],
        episodes=episodes
    )

    # 테스트 및 비디오 생성
    test_reward = test_agent(agent, render=False, num_episodes=5)
    print(f"Final test reward (average of 5 episodes): {test_reward:.2f}")

    video_reward, steps = render_and_save_video(
        agent,
        f"mountaincar_ac_g{params['gamma']}_lrpi{params['lr_pi']}_lrv{params['lr_v']}.mp4"
    )

    return agent, rewards, test_reward


if __name__ == "__main__":
    # 하이퍼파라미터 탐색을 원한다면 아래 코드를 사용합니다
    print("Running hyperparameter search...")
    best_params, best_agent, results = hyperparameter_search()

    # 최적 파라미터로 최종 학습 및 비디오 생성
    final_agent, final_rewards, final_test_reward = final_training(best_params, episodes=500)

    # 결과 요약
    print("\nHyperparameter Search Results (Top 5):")
    for i, result in enumerate(results[:5]):
        print(
            f"{i + 1}. gamma={result['gamma']}, lr_pi={result['lr_pi']}, lr_v={result['lr_v']}: {result['avg_reward']:.2f}")

    print(
        f"\nBest Configuration: gamma={best_params['gamma']}, lr_pi={best_params['lr_pi']}, lr_v={best_params['lr_v']}")
    print(f"Final Test Reward: {final_test_reward:.2f}")