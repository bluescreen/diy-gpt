
import numpy as np
import matplotlib.pylab as plt
import torch
from torch.nn import functional as F

xs = np.asarray([[-10], [-8], [-6], [-4], [-2], [0], [2], [4], [6], [8], [10]])
ys = 0.5 * xs + 7

xs = np.hstack((xs, np.ones([xs.shape[0], 1])))

xs = torch.tensor(xs).float()
ys = torch.tensor(ys).float()


ins = 1
outs = 1
nodes = 100
lr = 0.00001


def weights(ins, outs):
    ws = torch.randn(ins, outs)
    ws.requires_grad_(True)
    return ws


w0 = weights(ins+1, nodes)
w1 = weights(nodes, nodes)
w2 = weights(nodes, outs)

optimizer = torch.optim.SGD([w0, w1, w2], lr)
ers = []

for i in range(5000):
    x0 = xs

    z0 = (x0 @ w0)
    x1 = torch.sin(z0)
    z1 = (x1 @ w1)
    x2 = torch.sin(z1)
    yh = (x2 @ w2)

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
plt.show()
