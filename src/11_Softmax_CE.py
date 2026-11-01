import numpy as np
import torch
from torch import nn


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x_np = np.array([2.0, 1.0, 0.1])
print('softmax numpy :', softmax(x_np))

x = torch.tensor([2.0, 1.0, 0.1])
print('softmax torch :', torch.softmax(x, dim=0))

def cross_entropy(actual, predicted):
    loss = -np.sun(actual * np.log(predicted))
    return loss

loss = nn.CrossEntropyLoss()

# 3 samples
Y = torch.tensor([2, 0, 1])
# nsamples x nclasses = 3x3
Y_pred_good = torch.tensor([[0.2, 1.0, 2.1], [2.0, 1.0, 0.1],[0.2, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.3, 1.5, 0.1],[0.3, 1.5, 0.1],[3.0, 0.2, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item(), l2.item())

_, prediction1 = torch.max(Y_pred_good, 0)
_, prediction2 = torch.max(Y_pred_bad, 0)
print(prediction1,prediction2)