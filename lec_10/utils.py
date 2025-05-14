import numpy as np
import matplotlib.pyplot as plt


def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = argmax(qs)  # OR np.argmax(qs)
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


# utils.py의 plot_total_reward 함수 수정
def plot_total_reward(reward_history):
    plt.figure(figsize=(12, 5))

    # 축 범위 계산 (두 그래프에 동일하게 적용하기 위함)
    x_max = len(reward_history)
    y_min = 0
    y_max = max(reward_history) + 10  # 여유 공간 추가

    # 첫 번째 그래프: 각 에피소드의 총 보상
    plt.subplot(1, 2, 1)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.xlim(0, x_max)
    plt.ylim(y_min, y_max)

    # 두 번째 그래프: 100회 평균한 총 보상
    plt.subplot(1, 2, 2)
    avg_rewards = []
    for i in range(len(reward_history)):
        if i < 100:
            avg_rewards.append(np.mean(reward_history[:i + 1]))
        else:
            avg_rewards.append(np.mean(reward_history[i - 99:i + 1]))

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.xlim(0, x_max)
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


