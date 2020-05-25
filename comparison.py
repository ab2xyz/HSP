import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import  nn as nn
import pandas as pd


def GetData(f):

    x=pd.read_csv(f).to_numpy()[:,1:]
    x=np.c_[x,np.zeros((x.shape[0],17*17-x.shape[1]))]


    y=x.copy()
    bn=nn.BatchNorm1d(289)
    y=bn(torch.Tensor(y)).detach().numpy()
    y=bn(torch.Tensor(y)).detach().numpy()
    y=bn(torch.Tensor(y)).detach().numpy()
    y1=y.mean(0).copy()
    # y1=y[0,:].copy()
    y1.resize((17,17))

    return y1


ySig=GetData(f='/home/i/IGSI/data/data/csv/train/T00_T00_45_ee/0000000_0001024.csv')
yBkg=GetData(f='/home/i/IGSI/data/data/csv/train/T00_DPM_45/0000000_0001024.csv')

ySig+=1e-6
yBkg+=1e-6


plt.figure('Comparison')
plt.subplot(121)
plt.title('sig')
plt.imshow(ySig)
plt.subplot(122)
plt.title('bkg')
plt.imshow(yBkg)


plt.figure('Comparison ratio')
ratioSigBkg=ySig/yBkg


plt.imshow(ratioSigBkg)

# print(ySig/yBkg)
# print()



plt.show()

#
