#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self,numInput=232, numClass=34):
        super(DNN, self).__init__()
        self.numClass=numClass


        self.layer1=nn.Sequential(
            nn.Linear(numInput,numInput),
            # nn.BatchNorm1d(240),
            nn.ReLU(True),
            nn.Dropout(0.15))

        self.layer2=nn.Sequential(
            nn.Linear(numInput,numInput),
            # nn.BatchNorm1d(240),
            nn.ReLU(True),
            nn.Dropout(0.15))

        self.layer3=nn.Sequential(
            nn.Linear(numInput,self.numClass))

    def forward(self, x):

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        x=x.view(-1,self.numClass)


        return x





# Residual block
class ResidualDNNBlock(nn.Module):
    def __init__(self, numNeuronPerLayer):
        super(ResidualDNNBlock, self).__init__()

        self.linear1 = nn.Linear(numNeuronPerLayer,numNeuronPerLayer)
        self.bn1 = nn.BatchNorm1d(numNeuronPerLayer)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(numNeuronPerLayer,numNeuronPerLayer)
        self.bn2 = nn.BatchNorm1d(numNeuronPerLayer)


    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out




# ResidualDNNBlock
class ResidualDNN(nn.Module):
    def __init__(self, block=ResidualDNNBlock,numBlocks=10, numInput=232, numClass=34):
        super(ResidualDNN, self).__init__()


        self.linear = nn.Linear(numInput,numInput)
        self.bn = nn.BatchNorm1d(numInput)
        self.relu = nn.ReLU(inplace=True)
        self.layerBlock = self.makeLayer(block, numBlocks, numInput)

        self.fc = nn.Linear(numInput, numClass)

    def makeLayer(self, block, numBlocks, numNeuronPerLayer):
        layers = []
        for i in range(numBlocks):
            layers.append(block(numNeuronPerLayer))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layerBlock(out)
        out = self.fc(out)
        return out








class ResDNN(nn.Module):
    def __init__(self,numInput=232, numClass=34):
        super(ResDNN, self).__init__()
        self.numClass=numClass


        self.layer0=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer1=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer2=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer3=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer4=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer5=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer6=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer7=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer8=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer9=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer10=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer11=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer12=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer13=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer14=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput))

        self.layer15=nn.Sequential(
            nn.Linear(numInput,self.numClass))

        self.relu=nn.ReLU(True)

    def forward(self, x):
        I=x
        x=self.layer0(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer1(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer2(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer3(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer4(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer5(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer6(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer7(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer8(x)
        x+=I
        x=self.relu(x)

        I=x
        x=self.layer9(x)
        x+=I
        x=self.relu(x)


        I=x
        x=self.layer10(x)
        x+=I
        x=self.relu(x)


        I=x
        x=self.layer11(x)
        x+=I
        x=self.relu(x)


        I=x
        x=self.layer12(x)
        x+=I
        x=self.relu(x)


        I=x
        x=self.layer13(x)
        x+=I
        x=self.relu(x)


        I=x
        x=self.layer14(x)
        x+=I
        x=self.relu(x)


        x=self.layer15(x)

        x=x.view(-1,self.numClass)


        return x




if __name__=='__main__':

    numInput=232
    numClass=2
    oDNN=DNN(numInput=numInput, numClass=numClass)

    import torch
    xIn=torch.randn(4,numInput)

    xOut=oDNN(xIn)

    print(xIn)
    print(xOut)
    print(xIn.shape)
    print(xOut.shape)




    #
