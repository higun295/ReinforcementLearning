import sys
sys.path.append("../src/")
import wandb
import os
from config import *
from pong_wrapper import *
from process_image import *
from utilities import *
from network import *
from agent import *
import matplotlib.pyplot as plt

wandb.init(
  project="tensorflow2_pong_{}".format(AGENT.lower()),
  tags=[AGENT.lower(), "CNN", "RL", "atari_pong"],
  config=CONFIG_WANDB,
)

pw = PongWrapper(ENV_NAME, history_length=4)
model = Model(num_actions=pw.env.action_space.n, hidden=HIDDEN)
agent = Agent(model)

def main():
    rewards_history = agent.train(pw)

    # 그래프 저장할 디렉토리 지정
    timestamp = time.strftime('%Y%m%d%H%M')
    save_dir = f"{PATH_SAVE_MODEL}/save_agent_{timestamp}"
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 그래프 그리기 및 저장
    plt.plot(rewards_history)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.grid(True)

    plot_path = os.path.join(plot_dir, "reward_plot.png")
    plt.savefig(plot_path)
    print(f"[✔] 리워드 그래프 저장됨: {plot_path}")

if __name__ == "__main__":
    try:
        main()
    finally:
        import time
        save_dir = f"{PATH_SAVE_MODEL}/save_agent_{time.strftime('%Y%m%d%H%M')}"
        model_path = f"{save_dir}/model.tf"
        video_path = f"{save_dir}/test_play.mp4"

        agent.save_model(model_path)
        reward = agent.test(pw, render=False, record_path=video_path)
        print(f"[✔] 저장 완료 - 모델: {model_path}, 영상: {video_path}, 리워드: {reward}")

