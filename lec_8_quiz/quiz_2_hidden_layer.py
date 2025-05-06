import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터셋 생성 - y = sin(4πx)
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(4 * np.pi * x) + np.random.rand(100, 1) * 0.1

lr = 0.1
iters = 2000
hidden_sizes = [10, 20, 50, 100, 200]


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


# 결과 저장용 리스트
results = []

# 각 은닉층 크기에 대한 학습 실행
for hidden_size in hidden_sizes:
    print(f"\n은닉층 크기 {hidden_size}로 학습 시작")

    # 모델과 옵티마이저 초기화
    model = TwoLayerNet(hidden_size, 1)
    optimizer = optimizers.Adam(lr)
    optimizer.setup(model)

    # 손실 기록
    loss_history = []

    # 학습 과정
    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()

        # 500번 반복마다 손실 기록
        if i % 500 == 0:
            loss_val = float(loss.data)
            loss_history.append(loss_val)
            print(f"반복 {i:5d}: 손실 = {loss_val:.6f}")

    # 최종 손실 출력
    final_loss = float(loss.data)
    print(f"은닉층 크기 {hidden_size}의 최종 손실: {final_loss:.6f}")

    # 예측값 생성
    t = np.arange(0, 1, .005)[:, np.newaxis]
    y_pred = model(t)

    # 결과 저장
    results.append({
        'hidden_size': hidden_size,
        'loss_history': loss_history,
        'final_loss': final_loss,
        'predictions': y_pred.data
    })

# 결과 시각화
plt.figure(figsize=(15, 10))

# 1. 손실 그래프 비교
plt.subplot(2, 2, 1)
for result in results:
    plt.plot(range(0, iters, 500), result['loss_history'],
             marker='o', linestyle='-', label=f"은닉층 크기 {result['hidden_size']} (최종 손실: {result['final_loss']:.4f})")

plt.xlabel('반복 횟수')
plt.ylabel('손실 (MSE)')
plt.title('은닉층 크기별 손실 변화')
plt.legend()
plt.grid(True)

# 2. 최종 손실 비교 (막대 그래프)
plt.subplot(2, 2, 2)
sizes = [result['hidden_size'] for result in results]
final_losses = [result['final_loss'] for result in results]

plt.bar(range(len(sizes)), final_losses, tick_label=[str(size) for size in sizes])
plt.xlabel('은닉층 크기')
plt.ylabel('최종 손실')
plt.title('은닉층 크기별 최종 손실 비교')
plt.grid(True, axis='y')

# 3. 원본 데이터와 실제 함수
plt.subplot(2, 2, 3)
plt.scatter(x, y, s=10, alpha=0.5, label='학습 데이터')
t_smooth = np.arange(0, 1, .001)[:, np.newaxis]
plt.plot(t_smooth, np.sin(4 * np.pi * t_smooth), 'g-', linewidth=2, label='실제 sin(4πx)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('원본 데이터와 실제 함수')
plt.legend()
plt.grid(True)

# 4. 모든 모델의 예측 결과 비교
plt.subplot(2, 2, 4)
t = np.arange(0, 1, .005)[:, np.newaxis]
plt.scatter(x, y, s=10, alpha=0.3, label='학습 데이터')
plt.plot(t_smooth, np.sin(4 * np.pi * t_smooth), 'g-', linewidth=1, alpha=0.7, label='실제 sin(4πx)')

for result in results:
    plt.plot(t, result['predictions'], linewidth=2, label=f"은닉층 크기 {result['hidden_size']}")

plt.xlabel('x')
plt.ylabel('y')
plt.title('은닉층 크기별 예측 결과 비교')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 예측 결과 개별 비교 (은닉층 크기별)
plt.figure(figsize=(15, 10))
t = np.arange(0, 1, .005)[:, np.newaxis]

for i, result in enumerate(results):
    plt.subplot(2, 3, i + 1)
    plt.scatter(x, y, s=10, alpha=0.3, label='학습 데이터')
    plt.plot(t_smooth, np.sin(4 * np.pi * t_smooth), 'g-', linewidth=1, alpha=0.7, label='실제 sin(4πx)')
    plt.plot(t, result['predictions'], 'r-', linewidth=2, label=f"은닉층 크기 {result['hidden_size']}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"은닉층 크기 {result['hidden_size']} (손실: {result['final_loss']:.4f})")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 최적 은닉층 크기 찾기
best_idx = np.argmin([r['final_loss'] for r in results])
best_result = results[best_idx]
print(f"\n최적 은닉층 크기: {best_result['hidden_size']}")
print(f"최종 손실: {best_result['final_loss']:.6f}")



