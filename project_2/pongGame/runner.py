import the_agent
import environment
import cv2
import numpy as np
import time

name = 'ALE/Pong-v5'

# 에이전트 생성 (학습 안하도록 설정)
agent = the_agent.Agent(possible_actions=[0, 2, 3], starting_mem_len=999999999, max_mem_len=750000,
                        starting_epsilon=0.0, learn_rate=0.00025)

# 저장된 가중치 로드
try:
    agent.model.load_weights('recent_weights.hdf5')
    print("✅ 모델 로드 완료!")
except:
    print("❌ recent_weights.hdf5 파일을 찾을 수 없습니다!")
    input("엔터를 눌러 종료...")
    exit()

# 환경을 human 모드로 생성
import gymnasium as gym

env = gym.make(name, render_mode="human", frameskip=4)

# 게임 속도 설정
SPEED_MULTIPLIER = 5.0  # 이 값을 조정해서 속도 변경 (높을수록 빠름)

print(f"🎮 게임 시작! (속도: {SPEED_MULTIPLIER}배)")
print("Ctrl+C로 게임 종료")

try:
    # 무한 게임 루프
    game_count = 0
    while True:
        game_count += 1
        print(f"\n=== Game {game_count} ===")

        obs, info = env.reset()
        score = 0
        done = False

        while not done:
            # 현재 상태로 액션 선택
            if len(agent.memory.frames) >= 4:
                state = [agent.memory.frames[-4], agent.memory.frames[-3], agent.memory.frames[-2],
                         agent.memory.frames[-1]]
                state = np.moveaxis(state, 0, 2) / 255
                state = np.expand_dims(state, 0)
                action = agent.get_action(state)
            else:
                action = 0  # 초기에는 기본 액션

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

            # 메모리에 추가
            processed_frame = environment.ppf.resize_frame(obs)
            agent.memory.add_experience(processed_frame, reward, action, done)

            # 화면 렌더링
            env.render()

            # 게임 속도 조절 (3배속으로 빠르게)
            time.sleep(0.005)  # 3배속 진행을 위한 최소 딜레이

        print(f"Score: {score}")

except KeyboardInterrupt:
    print("\n👋 게임을 종료합니다.")

env.close()