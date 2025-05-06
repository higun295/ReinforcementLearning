import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
import time


# 5x5 Grid World 환경 클래스
class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # 행동 공간(가능한 행동들)
        self.action_meaning = {  # 행동의 의미
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        # 보상 맵 설정 (5x5)
        # None은 벽(장애물), 1.0은 양의 보상, -1.0은 음의 보상
        self.reward_map = np.array([
            [0, 0, 0, -1.0, 1.0],
            [0, 0, 0, 0, 0],
            [0, None, None, 0, 0],
            [0, None, None, 0, 0],
            [0, 0, 0, 0, -1.0]
        ])

        self.goal_state = (0, 4)  # 목표 상태(사과 위치)
        self.wall_states = [(2, 1), (2, 2), (3, 1), (3, 2)]  # 2x2 장애물
        self.penalty_states = [(0, 3), (4, 4)]  # 패널티 상태(폭탄 위치)
        self.start_state = (4, 0)  # 시작 상태(좌표)
        self.agent_state = self.start_state  # 에이전트 초기 상태(좌표)

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                if self.reward_map[h, w] is not None:  # 벽이 아닌 상태만 반환
                    yield (h, w)

    def next_state(self, state, action):
        # 이동 위치 계산
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 이동한 위치가 그리드 월드의 테두리 밖이나 벽인가?
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif self.reward_map[ny, nx] is None:  # 벽 상태
            next_state = state

        return next_state  # 다음 상태 반환

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_q(self, q=None, print_value=True):
        """Q 테이블을 시각화하는 함수"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # 격자 그리기
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(color='black', linestyle='-', linewidth=1)

        # 장애물 그리기
        for wall_y, wall_x in self.wall_states:
            ax.add_patch(plt.Rectangle((wall_x, self.height - wall_y - 1), 1, 1, fc='gray'))

        # 보상과 패널티 그리기
        goal_y, goal_x = self.goal_state
        ax.add_patch(plt.Rectangle((goal_x, self.height - goal_y - 1), 1, 1, fc='green', alpha=0.3))
        ax.text(goal_x + 0.5, self.height - goal_y - 0.5, "+1", ha='center', va='center', fontsize=12)

        for penalty_y, penalty_x in self.penalty_states:
            ax.add_patch(plt.Rectangle((penalty_x, self.height - penalty_y - 1), 1, 1, fc='red', alpha=0.3))
            ax.text(penalty_x + 0.5, self.height - penalty_y - 0.5, "-1", ha='center', va='center', fontsize=12)

        # Q 값과 정책 그리기
        if q is not None:
            for state in self.states():
                y, x = state
                if state == self.goal_state:
                    continue

                # 각 행동에 대한 Q 값 가져오기
                q_values = [q.get((state, a), 0.0) for a in self.action_space]
                best_action = np.argmax(q_values)

                # 화살표로 최적 행동 표시
                arrows = ["↑", "↓", "←", "→"]
                arrow = arrows[best_action]

                # 화살표 위치 조정
                ax.text(x + 0.5, self.height - y - 0.5, arrow, ha='center', va='center', fontsize=20)

                # Q 값 표시
                if print_value:
                    for a, q_val in enumerate(q_values):
                        offsets = [(0.25, 0.8), (0.25, 0.2), (0.1, 0.5), (0.4, 0.5)]
                        offset_x, offset_y = offsets[a]
                        ax.text(x + offset_x, self.height - y - offset_y,
                                f"{q_val:.2f}", ha='center', va='center', fontsize=8)

        plt.title("5x5 Grid World Q-Values and Policy")
        plt.show()


# one-hot 인코딩 함수: 상태를 one-hot 벡터로 변환
def one_hot(state):
    HEIGHT, WIDTH = 5, 5
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    if 0 <= y < HEIGHT and 0 <= x < WIDTH:  # 유효한 상태 확인
        idx = WIDTH * y + x
        vec[idx] = 1.0
    return vec[np.newaxis, :]


# Q-Network 정의
class QNet(Model):
    def __init__(self):
        super().__init__()
        # 입력 차원: 5x5=25(상태 공간), 은닉층: 50, 출력 차원: 4(행동 공간)
        self.l1 = L.Linear(10)
        self.l2 = L.Linear(4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


# Q-Learning Agent 정의
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9  # 할인율
        self.lr = 0.01  # 학습률
        self.epsilon = 0.3  # 시작 탐험율을 높게 설정
        self.min_epsilon = 0.01  # 최소 탐험율 설정
        self.action_size = 4  # 행동 크기

        self.qnet = QNet()
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

        # 추가 디버깅용 변수
        self.steps_taken = 0
        self.successful_episodes = 0
        self.episode_lengths = []

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q.unchain()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        self.steps_taken += 1

        return loss.data


# 학습 파라미터 설정
episodes = 1000
epsilon_decay = 0.995  # 탐험율 감소 계수
max_steps_per_episode = 100  # 에피소드당 최대 스텝 수 제한
print_interval = 50  # 로그 출력 간격
save_interval = 500  # 모델 저장 간격

# 환경과 에이전트 초기화
env = GridWorld()
agent = QLearningAgent()

# 시간 측정 시작
start_time = time.time()

# 학습 진행
loss_history = []
episode_reward_history = []
steps_per_episode = []

try:
    for episode in range(episodes):
        state = env.reset()
        state_vec = one_hot(state)
        total_loss, total_reward, steps = 0, 0, 0
        done = False

        # 에피소드 내 스텝
        for step in range(max_steps_per_episode):
            action = agent.get_action(state_vec)
            next_state, reward, done = env.step(action)
            next_state_vec = one_hot(next_state)

            loss = agent.update(state_vec, action, reward, next_state_vec, done)

            total_loss += loss
            total_reward += reward
            steps += 1

            # 상태 업데이트
            state = next_state
            state_vec = next_state_vec

            # 목표 도달 또는 최대 스텝 초과 시 종료
            if done:
                agent.successful_episodes += 1
                break

        # 에피소드 기록
        loss_history.append(total_loss / steps if steps > 0 else 0)
        episode_reward_history.append(total_reward)
        steps_per_episode.append(steps)
        agent.episode_lengths.append(steps)

        # 탐험율 감소 (최소값 유지)
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * epsilon_decay)

        # 학습 과정 출력
        if (episode + 1) % print_interval == 0:
            elapsed_time = time.time() - start_time
            avg_loss = np.mean(loss_history[-print_interval:])
            avg_reward = np.mean(episode_reward_history[-print_interval:])
            avg_steps = np.mean(steps_per_episode[-print_interval:])
            success_rate = sum(
                [1 for s in steps_per_episode[-print_interval:] if s < max_steps_per_episode]) / print_interval

            print(f"Episode {episode + 1}/{episodes} [{elapsed_time:.1f}s]")
            print(f"  Loss: {avg_loss:.6f}, Reward: {avg_reward:.2f}, Steps: {avg_steps:.1f}")
            print(f"  Epsilon: {agent.epsilon:.4f}, Success Rate: {success_rate:.2f}")
            print(f"  Total Steps: {agent.steps_taken}, Total Success: {agent.successful_episodes}")

        # 학습 모델 주기적 저장 (선택사항)
        if (episode + 1) % save_interval == 0:
            # 이 시점에서 Q 테이블을 확인하거나 시각화할 수 있습니다
            print("\n---- 중간 Q 테이블 확인 ----")
            temp_Q = {}
            for state in env.states():
                state_vec = one_hot(state)
                for action in env.action_space:
                    qs = agent.qnet(state_vec)
                    temp_Q[(state, action)] = float(qs.data[0, action])

            # 최적 정책 텍스트로 출력
            print("\n현재 최적 정책:")
            for y in range(env.height):
                for x in range(env.width):
                    state = (y, x)
                    if env.reward_map[y, x] is None:  # 벽인 경우
                        print("  #  ", end="")
                    elif state == env.goal_state:
                        print("GOAL ", end="")
                    else:
                        q_values = [temp_Q.get((state, a), float('-inf')) for a in env.action_space]
                        best_action = np.argmax(q_values)
                        arrows = ["↑", "↓", "←", "→"]
                        print(f"  {arrows[best_action]}  ", end="")
                print()  # 줄바꿈

            # 배치 시각화는 주석 처리 (학습 중 중단 방지)
            # env.render_q(temp_Q)

except KeyboardInterrupt:
    print("\n학습 중단됨!")

# 총 학습 시간
total_time = time.time() - start_time
print(f"\n총 학습 시간: {total_time:.2f}초")

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
plt.plot(range(len(episode_reward_history)), episode_reward_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Reward')
plt.grid(True)

# 3. 스텝 수 그래프
plt.subplot(1, 3, 3)
plt.plot(range(len(steps_per_episode)), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.grid(True)

plt.tight_layout()
plt.show()

# 최종 Q 테이블 생성
Q = {}
for state in env.states():
    state_vec = one_hot(state)
    for action in env.action_space:
        qs = agent.qnet(state_vec)
        Q[(state, action)] = float(qs.data[0, action])

# Q 테이블과 최적 정책 시각화
env.render_q(Q)

# 최적 정책 출력
print("\n최종 최적 정책:")
for y in range(env.height):
    for x in range(env.width):
        state = (y, x)
        if env.reward_map[y, x] is None:  # 벽인 경우
            print("  #  ", end="")
        elif state == env.goal_state:
            print("GOAL ", end="")
        else:
            q_values = [Q.get((state, a), float('-inf')) for a in env.action_space]
            best_action = np.argmax(q_values)
            arrows = ["↑", "↓", "←", "→"]
            print(f"  {arrows[best_action]}  ", end="")
    print()  # 줄바꿈