#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch.nn as nn
import torch.nn.functional as F
class NN(nn.Module):
    def __init__(self,numClasses=34):
        super(NN, self).__init__()
        self.numClasses=numClasses

        self.conv0 = nn.Conv2d(1, 16, 3)
        self.conv1 = nn.Conv2d(16, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.conv6 = nn.Conv2d(16, 16, 3)
        self.conv7 = nn.Conv2d(16, self.numClasses, 3)

        self.bn0=nn.BatchNorm2d(16)
        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(16)
        self.bn3=nn.BatchNorm2d(16)
        self.bn4=nn.BatchNorm2d(16)
        self.bn5=nn.BatchNorm2d(16)
        self.bn6=nn.BatchNorm2d(16)





    def forward(self, x):
        x = self.bn0(F.relu(self.conv0(x)))
        x =self.bn1( F.relu(self.conv1(x)))
        x =self.bn2( F.relu(self.conv2(x)))
        x =self.bn3( F.relu(self.conv3(x)))
        x =self.bn4( F.relu(self.conv4(x)))
        x =self.bn5( F.relu(self.conv5(x)))
        x =self.bn6(F.relu(self.conv6(x)))
        x = self.conv7(x)

        x=x.view(-1,self.numClasses)


        return x


if __name__=='__main__':
    numClasses=2
    oNN=NN(numClasses)

    import torch
    xIn=torch.randn(4,1,17,17)

    xOut=oNN(xIn)


    print(xIn)
    print(xOut)
    print(xIn.shape)
    print(xOut.shape)




    #
