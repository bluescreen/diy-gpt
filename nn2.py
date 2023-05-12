
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.nn import functional as F

xs = np.asarray([[-10], [-8], [-6], [-4], [-2], [0], [2], [4], [6], [8], [10]])
ys = xs ** 2

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))

xs = torch.tensor(xs).float()
ys = torch.tensor(ys).float()


ins = 1
outs = 1
nodes = 200
lr = 0.003

params = []


def weights(ins, outs):
    ws = torch.randn(ins, outs) * 0.1
    ws.requires_grad_(True)
    params.append(ws)
    return ws


class Model():
    def __init__(self):
        self.w0 = weights(ins+1, nodes)
        self.w1 = weights(nodes, nodes)
        self.w2 = weights(nodes, outs)

    def forward(self, x):
        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)
        yh = (x @ self.w2)
        return yh


model = Model()
optimizer = torch.optim.Adam(params, lr)
ers = []

for i in range(5000):
    x0 = xs

    yh = model.forward(xs)

    loss = F.mse_loss(yh, ys)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    e = loss.item()
    if (i % 50 == 0):
        print(i, "Loss", e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(yh.detach())


value = -5
value = torch.tensor([value, 1]).float()
result = model.forward(value)

print(result)
plt.show()
