import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent

runs = 500
steps = 1000
epsilons = [0.1, 0.3, 0.01]
colors = ['blue', 'orange', 'green']
labels = ['0.1', '0.3', '0.01']

plt.figure(figsize=(8, 6))

for epsilon, color, label in zip(epsilons, colors, labels):
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.mean(all_rates, axis=0)
    plt.plot(avg_rates, label=f"{label}", color=color)

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.legend()
plt.grid(True)
plt.show()


