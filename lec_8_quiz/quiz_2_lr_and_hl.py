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
x = np.random.rand(100, 1)  # 0~1 사이의 무작위 입력값
y = np.sin(4 * np.pi * x) + np.random.rand(100, 1) * 0.1  # 노이즈 감소

# 하이퍼파라미터 설정
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]  # 학습률 변화
hidden_sizes = [10, 30, 50, 100, 200]  # 은닉층 크기 변화
iters = 2000  # 반복 횟수를 2000회로 줄임


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
all_results = []

# 모든 조합에 대한 실험 수행
for lr in learning_rates:
    for hidden_size in hidden_sizes:
        print(f"\n학습률 {lr}, 은닉층 크기 {hidden_size}로 학습 시작")

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

            # 400번 반복마다 손실 기록
            if i % 400 == 0:
                loss_val = float(loss.data)
                loss_history.append(loss_val)
                print(f"반복 {i:4d}: 손실 = {loss_val:.6f}")

        # 최종 손실 출력
        final_loss = float(loss.data)
        print(f"학습률 {lr}, 은닉층 크기 {hidden_size}의 최종 손실: {final_loss:.6f}")

        # 예측값 생성
        t = np.arange(0, 1, .01)[:, np.newaxis]
        y_pred = model(t)

        # 결과 저장
        all_results.append({
            'lr': lr,
            'hidden_size': hidden_size,
            'loss_history': loss_history,
            'final_loss': final_loss,
            'predictions': y_pred.data
        })

# 결과 분석 및 시각화

# 1. 학습률과 은닉층 크기에 따른 최종 손실 히트맵
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)

# 히트맵 데이터 준비
loss_matrix = np.zeros((len(learning_rates), len(hidden_sizes)))
for i, lr in enumerate(learning_rates):
    for j, hidden_size in enumerate(hidden_sizes):
        for result in all_results:
            if result['lr'] == lr and result['hidden_size'] == hidden_size:
                loss_matrix[i, j] = result['final_loss']
                break

plt.imshow(loss_matrix, cmap='viridis_r')
plt.colorbar(label='최종 손실')
plt.xticks(range(len(hidden_sizes)), [str(s) for s in hidden_sizes])
plt.yticks(range(len(learning_rates)), [str(r) for r in learning_rates])
plt.xlabel('은닉층 크기')
plt.ylabel('학습률')
plt.title('학습률과 은닉층 크기에 따른 최종 손실')

# 2. 학습률별 최종 손실 비교 (은닉층 크기 평균)
plt.subplot(2, 2, 2)
loss_by_lr = []
for lr in learning_rates:
    lr_losses = [r['final_loss'] for r in all_results if r['lr'] == lr]
    loss_by_lr.append(np.mean(lr_losses))

plt.bar(range(len(learning_rates)), loss_by_lr, tick_label=[str(lr) for lr in learning_rates])
plt.xlabel('학습률')
plt.ylabel('평균 최종 손실')
plt.title('학습률에 따른 평균 최종 손실')
plt.grid(True, axis='y')

# 3. 은닉층 크기별 최종 손실 비교 (학습률 평균)
plt.subplot(2, 2, 3)
loss_by_hidden = []
for hidden_size in hidden_sizes:
    hidden_losses = [r['final_loss'] for r in all_results if r['hidden_size'] == hidden_size]
    loss_by_hidden.append(np.mean(hidden_losses))

plt.bar(range(len(hidden_sizes)), loss_by_hidden, tick_label=[str(s) for s in hidden_sizes])
plt.xlabel('은닉층 크기')
plt.ylabel('평균 최종 손실')
plt.title('은닉층 크기에 따른 평균 최종 손실')
plt.grid(True, axis='y')

# 4. 최적 조합 찾기
best_idx = np.argmin([r['final_loss'] for r in all_results])
best_result = all_results[best_idx]

# 최적 조합의 예측 결과 시각화
plt.subplot(2, 2, 4)
t = np.arange(0, 1, .01)[:, np.newaxis]
t_smooth = np.arange(0, 1, .001)[:, np.newaxis]
plt.scatter(x, y, s=10, alpha=0.3, label='학습 데이터')
plt.plot(t_smooth, np.sin(4 * np.pi * t_smooth), 'g-', linewidth=1, alpha=0.7, label='실제 sin(4πx)')
plt.plot(t, best_result['predictions'], 'r-', linewidth=2,
         label=f"최적 조합: 학습률 {best_result['lr']}, 은닉층 크기 {best_result['hidden_size']}")
plt.xlabel('x')
plt.ylabel('y')
plt.title(f"최적 조합 (손실: {best_result['final_loss']:.4f})")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 학습률별 손실 변화 그래프 (은닉층 크기 30 고정)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
hidden_size_to_show = 30  # 은닉층 크기 30으로 고정

for lr in learning_rates:
    for result in all_results:
        if result['lr'] == lr and result['hidden_size'] == hidden_size_to_show:
            plt.plot(range(0, iters, 400), result['loss_history'],
                     marker='o', linestyle='-', label=f"학습률 {lr} (최종 손실: {result['final_loss']:.4f})")

plt.xlabel('반복 횟수')
plt.ylabel('손실 (MSE)')
plt.title(f'학습률별 손실 변화 (은닉층 크기 {hidden_size_to_show} 고정)')
plt.legend()
plt.grid(True)

# 은닉층 크기별 손실 변화 그래프 (학습률 0.1 고정)
plt.subplot(1, 2, 2)
lr_to_show = 0.1  # 학습률 0.1로 고정

for hidden_size in hidden_sizes:
    for result in all_results:
        if result['lr'] == lr_to_show and result['hidden_size'] == hidden_size:
            plt.plot(range(0, iters, 400), result['loss_history'],
                     marker='o', linestyle='-', label=f"은닉층 크기 {hidden_size} (최종 손실: {result['final_loss']:.4f})")

plt.xlabel('반복 횟수')
plt.ylabel('손실 (MSE)')
plt.title(f'은닉층 크기별 손실 변화 (학습률 {lr_to_show} 고정)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 최종 결과 요약 출력
print("\n==== 실험 결과 요약 ====")
print(f"반복 횟수: {iters}회")
print(f"가장 낮은 손실: {best_result['final_loss']:.6f}")
print(f"최적 조합: 학습률 {best_result['lr']}, 은닉층 크기 {best_result['hidden_size']}")

# 학습률의 영향과 은닉층 크기의 영향 비교
lr_influence = max(loss_by_lr) - min(loss_by_lr)
hidden_influence = max(loss_by_hidden) - min(loss_by_hidden)
print(f"\n학습률 변화에 따른 손실 변화폭: {lr_influence:.6f}")
print(f"은닉층 크기 변화에 따른 손실 변화폭: {hidden_influence:.6f}")
print(f"학습률의 영향이 은닉층 크기의 영향보다 {lr_influence / hidden_influence:.1f}배 큽니다.")