import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import  nn as nn
import pandas as pd

x=pd.read_csv('/home/i/IGSI/data/data/csv/train/T00_T00_45_ee/0000000_0001024.csv').to_numpy()[:,1:]
x=np.c_[x,np.zeros((x.shape[0],17*17-x.shape[1]))]

print(x.shape)

x1=x.mean(0).copy()
x1.resize((17,17))
plt.figure('x1')
plt.imshow(x1)


y=x.copy()
bn=nn.BatchNorm1d(289)
y=bn(torch.Tensor(y)).detach().numpy()
y=bn(torch.Tensor(y)).detach().numpy()
y1=y.mean(0).copy()
y1.resize((17,17))
plt.figure('y1')
plt.imshow(y1)


z=x.copy()
z=z.reshape(-1,1,17,17)
print(z.shape)
bnz=nn.BatchNorm2d(1)
z=bnz(torch.Tensor(z))
z1=z.mean(dim=[0,1]).detach().numpy()

print(z.shape)
print(z1.shape)

plt.figure('z1')
plt.imshow(z1)










plt.show()
#
