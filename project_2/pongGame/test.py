import the_agent
import environment
import cv2
import numpy as np
import gymnasium as gym
from datetime import datetime
import os

name = 'ALE/Pong-v5'

# 에이전트 생성
agent = the_agent.Agent(possible_actions=[0, 2, 3], starting_mem_len=999999999, max_mem_len=750000,
                        starting_epsilon=0.0, learn_rate=0.00025)

# 저장된 가중치 로드
if os.path.exists('recent_weights.hdf5'):
    agent.model.load_weights('recent_weights.hdf5')
    print("✅ 훈련된 모델 로드 완료!")
else:
    print("❌ 모델 파일을 찾을 수 없습니다. 랜덤 플레이로 진행합니다.")

# 환경을 rgb_array 모드로 생성 (비디오 저장용)
env = gym.make(name, render_mode="rgb_array")

# 첫 프레임으로 실제 크기 확인
obs, info = env.reset()
actual_height, actual_width = obs.shape[:2]
print(f"📐 실제 프레임 크기: {actual_width} x {actual_height}")

# 비디오 저장 설정
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
video_filename = f'pong_gameplay_{timestamp}.avi'  # AVI로 변경 (더 안정적)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID 코덱 사용
fps = 30  # 프레임률
frame_size = (actual_width, actual_height)  # 실제 크기 사용

video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

# VideoWriter 객체가 제대로 열렸는지 확인
if not video_writer.isOpened():
    print("❌ VideoWriter 초기화 실패!")
    print("다른 코덱을 시도합니다...")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG 코덱 시도
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    if not video_writer.isOpened():
        print("❌ 비디오 저장을 할 수 없습니다.")
        exit(1)

print(f"✅ VideoWriter 초기화 성공!")
print(f"📐 저장 크기: {frame_size}")
print(f"🎞️ 코덱: {fourcc}")
env.reset()  # 다시 리셋

print(f"🎬 비디오 저장 시작: {video_filename}")

env.reset()

# 게임 플레이 및 비디오 녹화
print(f"🎮 게임 플레이 시작!")

print(f"\n=== PONG Game ===")

obs, info = env.reset()
score = 0
step_count = 0
done = False

while not done:
    # 현재 상태로 액션 선택
    if len(agent.memory.frames) >= 4:
        state = [agent.memory.frames[-4], agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1]]
        state = np.moveaxis(state, 0, 2) / 255
        state = np.expand_dims(state, 0)
        action = agent.get_action(state)
    else:
        action = 0  # 초기에는 기본 액션

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    score += reward
    step_count += 1

    # 메모리에 추가 (상태 관리용)
    processed_frame = environment.ppf.resize_frame(obs)
    agent.memory.add_experience(processed_frame, reward, action, done)

    # 프레임을 비디오에 저장
    frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 사용
    video_writer.write(frame_bgr)

    # 진행 상황 출력 (100 스텝마다)
    if step_count % 100 == 0:
        action_names = {0: "NOOP", 2: "RIGHT", 3: "LEFT"}
        print(f"  Step {step_count:4d} | Score: {score:3.0f} | Action: {action_names.get(action, action)}")

    # 점수 변화가 있을 때 출력
    if reward != 0:
        if reward > 0:
            print(f"  🎯 SCORE! +{reward} (Total: {score})")
        else:
            print(f"  😞 LOST! {reward} (Total: {score})")

print(f"게임 완료 | 최종 점수: {score} | 총 스텝: {step_count}")

# 비디오 저장 완료
video_writer.release()
env.close()

print(f"\n🎬 비디오 저장 완료: {video_filename}")
print(f"📊 게임 플레이 영상이 저장되었습니다.")

# 비디오 파일 정보 출력
if os.path.exists(video_filename):
    file_size = os.path.getsize(video_filename) / (1024 * 1024)  # MB 단위
    print(f"📁 파일 크기: {file_size:.1f} MB")
    print(f"🎞️ 해상도: {frame_size[0]} x {frame_size[1]}")
    print(f"⏱️ FPS: {fps}")

    # 재생 방법 안내
    print(f"\n▶️ 재생 방법:")
    print(f"   - VLC Player, Windows Media Player 등으로 재생 가능")
    print(f"   - 또는 코드에서: cv2.VideoCapture('{video_filename}')로 읽기 가능")
else:
    print("❌ 비디오 파일 저장에 실패했습니다.")