
import numpy as np
import matplotlib.pylab  as plt
import torch
from torch.nn import functional as F

with open("data.txt", "r", encoding='utf-8') as f:
    text = f.read()

text = text.lower()
chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
data = [stoi[c] for c in text]
vocab_size = len(chars)

ins = 5
outs = 1
nodes = 200
lr = 0.003


data = torch.tensor(data).float()

params = []
def weights(ins, outs):
    ws = torch.randn(ins, outs) * 0.1
    ws.requires_grad_(True)
    params.append(ws)
    return ws

class Model():
    def __init__(self):
        self.w0 = weights(ins, nodes)
        self.w1 = weights(nodes, nodes)
        self.w2 = weights(nodes, outs)

    def forward(self, x):
        x = torch.relu(x @self.w0)
        x = torch.relu(x @self.w1)
        yh = (x @ self.w2)
        return yh

model = Model()
optimizer = torch.optim.Adam(params, lr)
ers = []

for i in range(5000):
    b = torch.randint(len(data) - ins, (100,))
    xs = torch.stack([data[i:i+ins] for i in b])
    ys = torch.stack([data[i+ins:i+ins+1] for i in b])

    yh = model.forward(xs)

    loss = F.mse_loss(yh, ys)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    e = loss.item()
    if(i % 500 == 0):
        print("Loss", e)
    ers.append(e)

plt.figure(1)
plt.plot(ers)

plt.figure(2)
plt.plot(ys)
plt.plot(yh.detach())



plt.show()
