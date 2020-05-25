import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import  nn as nn


import cv2
im = cv2.imread("yinyang.png")
print(type(im))
print(im.shape)


# plt.imshow(im)

x=im.transpose(2,0,1)
print(x.shape)

x=x+(np.random.random(x.shape)*100-50)//1

x0=x.mean(axis=0)
plt.imshow(x0)

x=x[np.newaxis,:]



print(x.shape)

X=torch.Tensor(x)

conv = nn.Conv2d(3, 3, 13,padding=6)
bn = nn.BatchNorm2d(3)

def XPlot(X):
    y=X.mean(dim=[0,1])
    return y.detach().numpy()


n=1

fig=plt.figure('fig')
for i in range(n):
    X=conv(X)
    plt.clf()
    plt.imshow(XPlot(X))
    plt.title(str(i)+'conv')
    plt.pause(1)

    print(X.shape)

    X=bn(X)
    plt.clf()
    plt.imshow(XPlot(X))
    plt.title(str(i)+'bn')
    plt.pause(1)

y0=XPlot(X)
fig2=plt.figure('ratio')
plt.imshow(y0/x0)





plt.show()
#
