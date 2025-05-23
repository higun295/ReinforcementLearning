import matplotlib.pyplot as plt
import numpy as np

# paste.txt 파일에서 추출한 10의 배수 에피소드 데이터 (0, 10, 20, 30, ..., 990)
known_episodes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                  200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
                  400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
                  600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790,
                  800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990]

known_rewards = [
    -76.83397683397669, -71.53024911032071, -11.66077738515968, 176.59574468085532, -27.899686520376743,
    -24.242424242424907, -57.14285714285765, -67.85714285714324, -8.127208480565166, -36.36363636363688,
    -70.5882352941176, -58.95522388059751, -63.696369636964235, -48.14814814814876, -63.503649635037135,
    -71.51898734177229, -54.225352112676646, -73.2441471571907, -47.882736156352365, -74.44089456869007,
    -63.23529411764766, -75.43859649122805, -69.23076923076975, -32.43243243243271, -70.41420118343252,
    -82.07885304659462, 4.693140794223547, -34.02777777777865, -90.56603773584818, -28.571428571428605,
    -32.43243243243293, -90.19607843137169, -26.573426573427245, -80.9523809523807, -37.062937062937195,
    -55.47945205479526, -60.28880866426061, 28.37837837837747, -51.12781954887287, -60.854092526691396,
    -27.83505154639247, -34.615384615384784, -76.43097643097634, -86.30136986301304, -85.50724637681105,
    -31.97278911564652, -46.61921708185122, -26.17449664429601, -66.66666666666708, -19.230769230769777,
    -71.24600638977653, -67.85714285714359, -85.07462686567118, -30.23255813953488, 3.857566765579281,
    -19.1176470588243, -12.07142857142641, -63.57615894039796, -10.216718266254336, 61.870503597123744,
    23.674911660776644, -15.492957746479291, 85.60606060606159, -15.441176470588815, -32.65993265993303,
    -57.792207792208316, -75.69444444444439, -89.61937716262892, -83.38870431893639, 71.42857142857537,
    -83.76623376623327, -47.73519163763135, -78.91566265060237, -37.72893772893786, -82.45614035087675,
    -24.812030075188986, -25.781250000000735, -85.86572438162487, -6.6265060240966545, -65.83850931677065,
    -38.05309734513356, -80.98859315589317, -79.59183673469371, -70.80291970802958, 1.1750798722059272,
    -3.225806451613714, -69.81132075471747, -74.52229299363057, -1.9607843137258274, -67.625899280576,
    -14.47368421052705, -83.92282958199306, -64.66431095406445, -25.423728813560135, -81.30841121495295,
    -82.87671232876673, -65.87030716723599, -51.89003436426173, -78.02197802197767, -18.727915194346995
]

# myplot.png 그래프를 참고하여 1000개의 전체 데이터 생성
np.random.seed(42)  # 재현 가능한 결과

full_rewards = []

for episode in range(1000):
    if episode in known_episodes:
        # 알려진 데이터는 그대로 사용
        idx = known_episodes.index(episode)
        full_rewards.append(known_rewards[idx])
    else:
        # 중간 데이터는 그래프 패턴을 보고 생성
        # 가장 가까운 알려진 데이터 찾기
        before_idx = -1
        after_idx = -1

        for i, known_ep in enumerate(known_episodes):
            if known_ep < episode:
                before_idx = i
            elif known_ep > episode and after_idx == -1:
                after_idx = i
                break

        # 보간을 위한 기준값 계산
        if before_idx >= 0 and after_idx >= 0:
            # 두 알려진 값 사이의 선형 보간 + 노이즈
            before_reward = known_rewards[before_idx]
            after_reward = known_rewards[after_idx]
            before_episode = known_episodes[before_idx]
            after_episode = known_episodes[after_idx]

            # 선형 보간
            ratio = (episode - before_episode) / (after_episode - before_episode)
            base_reward = before_reward + (after_reward - before_reward) * ratio

        elif before_idx >= 0:
            # 마지막 구간
            base_reward = known_rewards[before_idx]
        else:
            # 첫 구간
            base_reward = known_rewards[0]

        # 그래프의 변동성을 반영한 노이즈 추가
        # myplot.png를 보면 대부분 -100~50 사이에 있고, 가끔 큰 양수 스파이크가 있음
        noise_std = 20  # 기본 노이즈
        noise = np.random.normal(0, noise_std)

        # 가끔 큰 성공 스파이크 (그래프에서 보이는 큰 양수값들)
        if np.random.random() < 0.05:  # 5% 확률로 큰 성공
            spike = np.random.uniform(100, 250)
            reward = spike + noise * 0.5
        else:
            reward = base_reward + noise

        # 값의 범위 제한 (myplot.png 그래프 패턴에 맞게 y축 범위 고려)
        reward = np.clip(reward, -100, 200)  # y축 범위에 맞게 조정

        full_rewards.append(reward)


# 100회 이동평균 계산
def calculate_moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data)):
        if i < window_size - 1:
            # 처음에는 가능한 데이터만 사용
            avg = np.mean(data[:i + 1])
        else:
            # 100개 윈도우의 평균
            avg = np.mean(data[i - window_size + 1:i + 1])
        moving_averages.append(avg)
    return moving_averages


# 100회 이동평균 계산
moving_avg_100 = calculate_moving_average(full_rewards, 100)

# myplot.png와 동일한 스타일로 그래프 그리기
plt.figure(figsize=(8, 6))  # myplot.png와 유사한 크기
plt.plot(range(1000), full_rewards, color='blue', linewidth=0.8)  # myplot.png와 같은 파란색
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.ylim(-100, 200)  # myplot.png와 동일한 y축 범위
plt.xlim(0, 1000)  # x축 범위도 명확히 설정
plt.grid(True)
plt.tight_layout()
plt.show()

# 100회 평균 그래프
plt.figure(figsize=(8, 6))
plt.plot(range(1000), moving_avg_100, color='red', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Average Total Reward (100 episodes)')
plt.ylim(-100, 200)  # 동일한 y축 범위 적용
plt.xlim(0, 1000)
plt.grid(True)
plt.tight_layout()
plt.show()

# 결합된 그래프
plt.figure(figsize=(8, 6))
plt.plot(range(1000), full_rewards, color='blue', linewidth=0.8, alpha=0.6, label='Episode Reward')
plt.plot(range(1000), moving_avg_100, color='red', linewidth=2, label='100-Episode Moving Average')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.ylim(-100, 200)  # y축 범위 제한
plt.xlim(0, 1000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 통계 정보 출력
print(f"전체 평균 보상: {np.mean(full_rewards):.2f}")
print(f"최종 100회 평균 보상: {np.mean(full_rewards[-100:]):.2f}")
print(f"최고 보상: {np.max(full_rewards):.2f}")
print(f"최저 보상: {np.min(full_rewards):.2f}")
print(f"학습 초기 100회 평균: {np.mean(full_rewards[:100]):.2f}")
print(f"학습 후기 100회 평균: {np.mean(full_rewards[-100:]):.2f}")

# 검증: 알려진 데이터 포인트들이 제대로 반영되었는지 확인
print("\n알려진 데이터 포인트 검증:")
for i in [0, 30, 100, 500, 990]:
    if i in known_episodes:
        idx = known_episodes.index(i)
        print(f"Episode {i}: 원본 {known_rewards[idx]:.2f}, 생성된 데이터 {full_rewards[i]:.2f}")