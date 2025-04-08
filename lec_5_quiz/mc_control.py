import numpy as np
from collections import defaultdict
from gridworld import GridWorld
import random

class McAgent:
    def __init__(self, env, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = defaultdict(lambda: 0.0)
        self.policy = defaultdict(lambda: {a: 0.25 for a in env.actions()})

    def select_action(self, state):
        probs = self.policy[state]
        actions = list(probs.keys())
        probs_values = list(probs.values())
        return np.random.choice(actions, p=probs_values)

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def update(self, episode):
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                self.Q[(state, action)] += self.alpha * (G - self.Q[(state, action)])

                # policy 개선 (epsilon-greedy)
                qs = [self.Q[(state, a)] for a in self.env.actions()]
                max_q = max(qs)
                max_actions = [a for a, q in zip(self.env.actions(), qs) if q == max_q]
                probs = {a: self.epsilon / len(self.env.actions()) for a in self.env.actions()}
                for a in max_actions:
                    probs[a] += (1 - self.epsilon) / len(max_actions)

                self.policy[state] = probs

if __name__ == '__main__':
    env = GridWorld()
    agent = McAgent(env, epsilon=0.4, alpha=0.5)

    for episode in range(5000):
        ep = agent.generate_episode()
        agent.update(ep)

    env.render_q(agent.Q)
    env.render_v(None, agent.policy)

