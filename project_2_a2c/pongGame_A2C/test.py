from pong_wrapper import PongWrapper
from network import Model
from agent import Agent
from config import *

pw = PongWrapper(ENV_NAME, render_mode="human")
model = Model(num_actions=pw.env.action_space.n, hidden=HIDDEN)
agent = Agent(model)

reward = agent.test(pw, render=True)
print(f"[✔] 리워드: {reward}")