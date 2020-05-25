import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import  nn as nn
import pandas as pd


def GetData(f):

    x=pd.read_csv(f).to_numpy()[:,1:]
    x=np.c_[x,np.zeros((x.shape[0],17*17-x.shape[1]))]

    return x

ySig=GetData(f='/home/i/IGSI/data/data/csv/train/T00_T00_45_ee/0000000_0001024.csv')
yBkg=GetData(f='/home/i/IGSI/data/data/csv/train/T00_DPM_45/0000000_0001024.csv')

# ySig+=1e-6
# yBkg+=1e-6

nSig=ySig.shape[0]
nBkg=yBkg.shape[0]

data=np.r_[ySig,yBkg]

print(data.shape)

y=data.copy()
bn=nn.BatchNorm1d(289)
y=bn(torch.Tensor(y)).detach().numpy()
y=bn(torch.Tensor(y)).detach().numpy()
y=bn(torch.Tensor(y)).detach().numpy()

zSig=y[:nSig,:]
zBkg=y[nSig:,:]


print(zSig.shape,zBkg.shape)

zSigMean=zSig.mean(0)
zBkgMean=zBkg.mean(0)

zSigMeanPlot=zSigMean.reshape(17,17)
zBkgMeanPlot=zBkgMean.reshape(17,17)



plt.figure('Comparison')
plt.subplot(121)
plt.title('sig')
plt.imshow(zSigMeanPlot)
plt.subplot(122)
plt.title('bkg')
plt.imshow(zBkgMeanPlot)

zSigMeanPlot+=1e-6
zBkgMeanPlot+=1e-6
plt.figure('Comparison ratio')
ratioSigBkg=zSigMeanPlot/zBkgMeanPlot
plt.imshow(zSigMeanPlot)
plt.title('sig/bkg')




plt.show()



#
