import the_agent
import environment
import matplotlib.pyplot as plt
import time
from collections import deque, Counter
import numpy as np
import wandb
import os
from datetime import datetime

# wandb 초기화
wandb.init(
   project="pong-dqn",
   config={
       "game": "ALE/Pong-v5",
       "possible_actions": [0, 2, 3],
       "starting_mem_len": 50000,
       "max_mem_len": 750000,
       "starting_epsilon": 0.1,
       "learn_rate": 0.00025,
       "gamma": 0.95,
       "epsilon_decay": 0.9 / 100000,
       "epsilon_min": 0.05,
       "batch_size": 32,
       "target_update_freq": 10000,
       "save_freq": 50000,
       "episodes": 30
   }
)

name = 'ALE/Pong-v5'

agent = the_agent.Agent(
   possible_actions=[0, 2, 3],
   starting_mem_len=50000,
   max_mem_len=750000,
   starting_epsilon=0.1,
   learn_rate=0.00025
)

# 기존 가중치 로드
if os.path.exists('recent_weights.hdf5'):
   agent.model.load_weights('recent_weights.hdf5')
   agent.model_target.load_weights('recent_weights.hdf5')
   print("기존 가중치 로드 완료!")
else:
   print("기존 가중치 파일이 없습니다. 새로 학습을 시작합니다.")

env = environment.make_env(name, agent)

# 분석을 위한 변수들
last_100_avg = [-21]
scores = deque(maxlen=100)
max_score = -21
total_steps = 0
episode_rewards = []
episode_actions = []
loss_values = []

# 로그 파일 생성
log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_filename, 'w', encoding='utf-8') as f:
   f.write("=== PONG DQN 학습 로그 ===\n")
   f.write(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
   f.write("=" * 50 + "\n\n")

env.reset()

for i in range(30):
   episode_start_time = time.time()
   timesteps_before = agent.total_timesteps

   # 에피소드별 분석 변수 초기화
   episode_action_counts = Counter()
   episode_reward_sum = 0
   positive_rewards = 0
   negative_rewards = 0
   exploration_steps = 0

   # 에피소드 실행
   score = environment.play_episode(name, env, agent, debug=False)

   episode_steps = agent.total_timesteps - timesteps_before
   episode_duration = time.time() - episode_start_time
   total_steps += episode_steps

   scores.append(score)
   if score > max_score:
       max_score = score

   avg_score_100 = sum(scores) / len(scores) if len(scores) > 0 else score

   # 메모리에서 최근 에피소드 데이터 분석
   if len(agent.memory.actions) > episode_steps:
       recent_actions = list(agent.memory.actions)[-episode_steps:]
       recent_rewards = list(agent.memory.rewards)[-episode_steps:]

       # 액션 분포 계산
       episode_action_counts = Counter(recent_actions)

       # 리워드 분석
       for reward in recent_rewards:
           episode_reward_sum += reward
           if reward > 0:
               positive_rewards += 1
           elif reward < 0:
               negative_rewards += 1

       # 탐험 비율 계산 (epsilon 기반 추정)
       exploration_steps = int(episode_steps * agent.epsilon)

   # Q-값 분석 (현재 상태 기준)
   q_values_mean = 0
   q_values_std = 0
   if len(agent.memory.frames) >= 4:
       state = [agent.memory.frames[-4], agent.memory.frames[-3],
                agent.memory.frames[-2], agent.memory.frames[-1]]
       state = np.moveaxis(state, 0, 2) / 255
       state = np.expand_dims(state, 0)
       q_values = agent.model.predict(state, verbose=0)[0]
       q_values_mean = float(np.mean(q_values))
       q_values_std = float(np.std(q_values))

   # wandb 로그 기록 (확장된 내용)
   log_data = {
       "episode": i,
       "score": score,
       "max_score": max_score,
       "avg_score_100": avg_score_100,
       "episode_steps": episode_steps,
       "total_timesteps": agent.total_timesteps,
       "episode_duration": episode_duration,
       "epsilon": agent.epsilon,
       "learns_count": agent.learns,
       "memory_size": len(agent.memory.frames),

       # 추가 분석 항목들
       "q_value_mean": q_values_mean,
       "q_value_std": q_values_std,
       "positive_rewards": positive_rewards,
       "negative_rewards": negative_rewards,
       "episode_reward_sum": episode_reward_sum,
       "action_0_count": episode_action_counts.get(0, 0),
       "action_2_count": episode_action_counts.get(2, 0),
       "action_3_count": episode_action_counts.get(3, 0),
       "exploration_ratio": exploration_steps / episode_steps if episode_steps > 0 else 0,
       "memory_usage_ratio": len(agent.memory.frames) / agent.memory.max_len,
       "steps_per_second": episode_steps / episode_duration if episode_duration > 0 else 0
   }

   wandb.log(log_data)

   # 콘솔 출력
   print(f'\n=== Episode: {i} ===')
   print(f'Score: {score} | Max: {max_score} | Avg(100): {avg_score_100:.2f}')
   print(f'Steps: {episode_steps} | Duration: {episode_duration:.2f}s')
   print(f'Epsilon: {agent.epsilon:.4f} | Learns: {agent.learns}')
   print(f'Q-values: μ={q_values_mean:.3f}, σ={q_values_std:.3f}')
   print(
       f'Actions: 0={episode_action_counts.get(0, 0)}, 2={episode_action_counts.get(2, 0)}, 3={episode_action_counts.get(3, 0)}')
   print(f'Rewards: +{positive_rewards}, -{negative_rewards}, Σ={episode_reward_sum}')

   # 텍스트 파일에 로그 기록
   with open(log_filename, 'a', encoding='utf-8') as f:
       f.write(f"Episode {i:3d} | Score: {score:6.1f} | Steps: {episode_steps:4d} | ")
       f.write(f"Epsilon: {agent.epsilon:.4f} | Q_μ: {q_values_mean:6.3f} | ")
       f.write(
           f"Actions(0/2/3): {episode_action_counts.get(0, 0):2d}/{episode_action_counts.get(2, 0):2d}/{episode_action_counts.get(3, 0):2d} | ")
       f.write(f"Rewards(+/-): {positive_rewards:2d}/{negative_rewards:2d}\n")

   # 100 에피소드마다 그래프 저장
   if i % 100 == 0 and i != 0:
       last_100_avg.append(avg_score_100)
       plt.figure(figsize=(10, 6))
       plt.plot(np.arange(0, i + 1, 100), last_100_avg)
       plt.title('Average Score over 100 Episodes')
       plt.xlabel('Episodes')
       plt.ylabel('Average Score')
       plt.grid(True)
       plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
       wandb.log({"training_progress": wandb.Image('training_progress.png')})
       plt.close()

# 학습 완료 후 모델 저장
print("\n🎯 학습 완료! 최종 모델 저장 중...")

# 타임스탬프로 최종 모델 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
final_model_name = f'pong_dqn_final_{timestamp}.hdf5'
agent.model.save_weights(final_model_name)
print(f"✅ 최종 모델 저장 완료: {final_model_name}")

# 최고 성능 모델도 저장 (선택사항)
best_model_name = f'pong_dqn_best_{max_score}pts_{timestamp}.hdf5'
agent.model.save_weights(best_model_name)
print(f"✅ 최고 성능 모델 저장 완료: {best_model_name}")

# 최종 요약 통계
final_stats = {
   "final_max_score": max_score,
   "final_avg_score": avg_score_100,
   "total_episodes": i + 1,
   "total_timesteps": agent.total_timesteps,
   "total_learns": agent.learns,
   "final_epsilon": agent.epsilon,
   "final_memory_size": len(agent.memory.frames),
   "final_model_saved": True,
   "final_model_name": final_model_name
}

wandb.log(final_stats)

# 텍스트 파일에 최종 요약 기록
with open(log_filename, 'a', encoding='utf-8') as f:
   f.write("\n" + "=" * 50 + "\n")
   f.write("=== 최종 학습 결과 요약 ===\n")
   f.write(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
   f.write(f"총 에피소드: {i + 1}\n")
   f.write(f"최고 점수: {max_score}\n")
   f.write(f"평균 점수 (최근 100): {avg_score_100:.2f}\n")
   f.write(f"총 학습 스텝: {agent.total_timesteps}\n")
   f.write(f"총 학습 횟수: {agent.learns}\n")
   f.write(f"최종 Epsilon: {agent.epsilon:.4f}\n")
   f.write(f"메모리 사용량: {len(agent.memory.frames)} / {agent.memory.max_len}\n")
   f.write(f"최종 모델: {final_model_name}\n")
   f.write("=" * 50 + "\n")

print(f"\n✅ 학습 완료! 로그 파일이 저장되었습니다: {log_filename}")

wandb.finish()