#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch.nn as nn
import torch.nn.functional as F
class NN(nn.Module):
    def __init__(self,numClasses=34):
        super(NN, self).__init__()
        self.numClasses=numClasses


        self.layer1=nn.Sequential(
            nn.Linear(240,240),
            # nn.BatchNorm1d(240),
            nn.ReLU(True),
            nn.Dropout(0.15))

        self.layer2=nn.Sequential(
            nn.Linear(240,240),
            # nn.BatchNorm1d(240),
            nn.ReLU(True),
            nn.Dropout(0.15))

        self.layer3=nn.Sequential(
            nn.Linear(240,self.numClasses))

    def forward(self, x):

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

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
