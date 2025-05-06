import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000
hidden_size = 10
out_size = 1


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


def train_with_optimizer(optimizer_name, lr=0.2):
    model = TwoLayerNet(hidden_size, out_size)

    if optimizer_name == 'SGD':
        optimizer = optimizers.SGD(lr)
    elif optimizer_name == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr)
    elif optimizer_name == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr)
    elif optimizer_name == 'Adam':
        optimizer = optimizers.Adam(lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    optimizer.setup(model)

    loss_history = []

    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()

        if i % 1000 == 0:
            print(f"{optimizer_name} - Iteration {i}: Loss = {loss.data}")

        if i % 100 == 0:
            loss_history.append(loss.data)

    print(f"{optimizer_name} final loss: {loss.data}")
    return model, loss_history


optimizers_to_compare = ['SGD', 'MomentumSGD', 'AdaGrad', 'Adam']
models = {}
loss_histories = {}

for opt_name in optimizers_to_compare:
    print(f"\nTraining with {opt_name}...")
    models[opt_name], loss_histories[opt_name] = train_with_optimizer(opt_name)

plt.figure(figsize=(12, 6))
for opt_name in optimizers_to_compare:
    plt.plot(np.arange(0, iters, 100), loss_histories[opt_name], label=opt_name)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss comparison between optimizers')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 10))

t = np.arange(0, 1, .01)[:, np.newaxis]

for i, opt_name in enumerate(optimizers_to_compare):
    plt.subplot(2, 2, i + 1)
    plt.scatter(x, y, s=10, label='Data')
    y_pred = models[opt_name](t)
    plt.plot(t, y_pred.data, color='r', linewidth=2, label='Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{opt_name} result')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


