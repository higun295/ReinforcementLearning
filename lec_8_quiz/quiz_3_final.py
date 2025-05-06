import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from bigGridWorld.gridworld import GridWorld
import time
from dezero import Variable


# 5x5 Grid World용 one-hot 인코딩
def one_hot(state):
    HEIGHT, WIDTH = 5, 5
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(64)  # 은닉층 크기 증가 (기존 10에서 64로)
        self.l2 = L.Linear(32)  # 중간 은닉층 추가
        self.l3 = L.Linear(4)  # 출력층 (행동 수)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))  # 중간 층 활성화
        x = self.l3(x)
        return x


class QLearningAgent:
    def __init__(self):
        # 하이퍼파라미터 최적화
        self.gamma = 0.98  # 할인율 증가 (0.9 → 0.98)
        self.lr = 0.003  # 학습률 조정 (0.01 → 0.003)
        self.epsilon = 0.5  # 초기 탐험율 증가 (0.1 → 0.5)
        self.min_epsilon = 0.01  # 최소 탐험율
        self.decay_rate = 0.995  # 탐험율 감소 계수
        self.action_size = 4

        # 신경망 및 옵티마이저 초기화
        self.qnet = QNet()
        self.optimizer = optimizers.Adam(self.lr)  # SGD에서 Adam으로 변경
        self.optimizer.setup(self.qnet)

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        # 타겟 Q값 계산
        if done:
            next_q = np.zeros(1)  # 종료 상태면 다음 Q값은 0
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q.unchain()  # 그래디언트 계산에서 제외

        # 타겟 값은 보상 + 감마 * 다음 최대 Q값
        target = reward + self.gamma * next_q  # reward는 스칼라, next_q는 Variable

        # 현재 Q값 예측
        qs = self.qnet(state)
        q = qs[:, action]

        # 손실 계산 및 신경망 업데이트
        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data

    def update_epsilon(self):
        # 탐험율 감소 (최소값까지)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


# 학습 설정
env = GridWorld()
agent = QLearningAgent()
episodes = 2000  # 에피소드 증가
max_steps = 100  # 에피소드당 최대 스텝 수
print_interval = 100

# 학습 기록용 변수
loss_history = []
reward_history = []
success_history = []
steps_history = []

print("===== Q-Network 학습 시작 =====")
print(f"신경망 최적화 파라미터:")
print(f"  - 은닉층 구조: 64 -> 32 -> 4")
print(f"  - 학습률: {agent.lr}")
print(f"  - 할인율: {agent.gamma}")
print(f"  - 초기 탐험율: {agent.epsilon} (감소 계수: {agent.decay_rate}, 최소값: {agent.min_epsilon})")

# 시간 측정 시작
start_time = time.time()

try:
    for episode in range(episodes):
        state = env.reset()
        state_vec = one_hot(state)
        total_loss, total_reward, step_count = 0, 0, 0
        losses = []
        done = False

        # 에피소드 진행
        for step in range(max_steps):
            action = agent.get_action(state_vec)
            next_state, reward, done = env.step(action)
            next_state_vec = one_hot(next_state)

            loss = agent.update(state_vec, action, reward, next_state_vec, done)
            if loss > 0:
                losses.append(loss)

            total_reward += reward
            step_count += 1

            state = next_state
            state_vec = next_state_vec

            if done:
                break

        # 에피소드 결과 기록
        loss_history.append(np.mean(losses) if losses else 0)
        reward_history.append(total_reward)
        success_history.append(1 if done else 0)
        steps_history.append(step_count)

        # 탐험율 감소
        agent.update_epsilon()

        # 진행 상황 출력
        if (episode + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            avg_success = np.mean(success_history[-print_interval:])
            avg_reward = np.mean(reward_history[-print_interval:])
            avg_steps = np.mean(steps_history[-print_interval:])

            print(f"Episode {episode + 1}/{episodes} [시간: {elapsed:.1f}초]")
            print(f"  평균 보상: {avg_reward:.2f}, 평균 스텝: {avg_steps:.1f}, 성공률: {avg_success:.2f}")
            print(f"  현재 탐험율: {agent.epsilon:.4f}")

except KeyboardInterrupt:
    print("\n학습 중단!")

# 총 학습 시간
total_time = time.time() - start_time
print(f"\n총 학습 시간: {total_time:.1f}초")

# 학습 결과 시각화
plt.figure(figsize=(15, 5))

# 1. 손실 그래프
plt.subplot(1, 3, 1)
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# 2. 보상 그래프
plt.subplot(1, 3, 2)
plt.plot(range(len(reward_history)), reward_history)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward')
plt.grid(True)

# 3. 성공률 그래프
plt.subplot(1, 3, 3)
window_size = 50
success_rates = []
for i in range(len(success_history)):
    if i < window_size:
        rate = np.mean(success_history[:i + 1])
    else:
        rate = np.mean(success_history[i - window_size + 1:i + 1])
    success_rates.append(rate)
plt.plot(range(len(success_rates)), success_rates)
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title(f'Success Rate (Window: {window_size})')
plt.grid(True)

plt.tight_layout()
plt.show()

# Q 테이블 생성
print("\nQ 테이블 생성 중...")
Q = {}
for state in env.states():
    if state in env.wall_states:
        continue
    state_vec = one_hot(state)
    for action in env.action_space:
        qs = agent.qnet(state_vec)
        Q[(state, action)] = float(qs.data[0, action])

# Q 테이블 시각화
print("\nQ 테이블 시각화:")
env.render_q(Q)

# 최적 정책 출력
print("\n최적 정책:")
for y in range(env.height):
    for x in range(env.width):
        state = (y, x)
        if state in env.wall_states:  # 벽인 경우
            print("  #  ", end="")
        elif state == env.goal_state:
            print("GOAL ", end="")
        else:
            q_values = [Q.get((state, a), float('-inf')) for a in env.action_space]
            best_action = np.argmax(q_values)
            arrows = ["↑", "↓", "←", "→"]
            print(f"  {arrows[best_action]}  ", end="")
    print()  # 줄바꿈

# 최적화 파라미터 요약
print("\n=== 신경망 최적화 파라미터 요약 ===")
print(f"1. 신경망 구조:")
print(f"   - 입력층: 25 (5x5 one-hot)")
print(f"   - 은닉층 1: 64 (ReLU)")
print(f"   - 은닉층 2: 32 (ReLU)")
print(f"   - 출력층: 4 (행동)")
print(f"2. 학습 파라미터:")
print(f"   - 옵티마이저: Adam(lr={agent.lr})")
print(f"   - 할인율(gamma): {agent.gamma}")
print(f"   - 탐험 전략: ε-greedy (초기 {0.5} → 최소 {agent.min_epsilon}, 감소율 {agent.decay_rate})")
print(f"3. 학습 결과:")
print(f"   - 에피소드: {episodes}")
print(f"   - 최종 성공률: {np.mean(success_history[-100:]):.2f}")
print(f"   - 평균 스텝 수: {np.mean(steps_history[-100:]):.1f}")