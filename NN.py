#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch.nn as nn
import torch.nn.functional as F
from numpy import prod as prod


class DNN_BN(nn.Module):
    def __init__(self,numInput=232, numClass=34):
        super(DNN_BN, self).__init__()
        self.numClass=numClass

        self.bn=nn.BatchNorm1d(numInput)

        self.layer1=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True))

        self.layer2=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True))

        self.layer3=nn.Sequential(
            nn.Linear(numInput,self.numClass))

    def forward(self, x):

        x=self.bn(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        x=x.view(-1,self.numClass)


        return x



class DNN_Dropout(nn.Module):
    def __init__(self,numInput=232, numClass=34):
        super(DNN_Dropout, self).__init__()
        self.numClass=numClass


        self.layer1=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.ReLU(True),
            nn.Dropout(0.15))

        self.layer2=nn.Sequential(
            nn.Linear(numInput,numInput),
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






class DNN_Res_Manual(nn.Module):
    def __init__(self,numInput=232, numClass=34):
        super(DNN_Res_Manual, self).__init__()
        self.numClass=numClass

        self.bn=nn.BatchNorm1d(numInput)

        self.layer0=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer1=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer2=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer3=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer4=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer5=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer6=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer7=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer8=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer9=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer10=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer11=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer12=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer13=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer14=nn.Sequential(
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput),
            nn.ReLU(True),
            nn.Linear(numInput,numInput),
            nn.BatchNorm1d(numInput))

        self.layer15=nn.Sequential(
            nn.Linear(numInput,self.numClass))

        self.relu=nn.ReLU(True)

    def forward(self, x):

        x=self.bn(x)

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





# Residual block
class Res_DNN_Block(nn.Module):
    def __init__(self, numNeuronPerLayer):
        super(Res_DNN_Block, self).__init__()

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




class Res_DNN(nn.Module):
    def __init__(self, block=Res_DNN_Block,numBlocks=10, numInput=232, numClass=34):
        super(Res_DNN, self).__init__()


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
        # x = self.linear(x)
        x = self.bn(x)
        # x = self.relu(x)
        x = self.layerBlock(x)
        x = self.fc(x)
        return x





class CNN(nn.Module):
    def __init__(self,numInput=[17,17],numClass=10):
        super(CNN, self).__init__()
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 8, 3)
        self.bn6 = nn.BatchNorm2d(8)


        self.fc1 = nn.Linear(8 * 5 * 5, 120)
        self.bnfc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bnfc2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, numClass)

    def forward(self, x):


        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))


        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.bnfc1(self.fc1(x)))
        x = F.relu(self.bnfc2(self.fc2(x)))
        x = self.fc3(x)
        return x




class CNN_BN(nn.Module):
    def __init__(self,numInput=[1,17,17],numClass=10):
        super(CNN_BN, self).__init__()
        self.bn = nn.BatchNorm1d(289)


        self.conv1 = nn.Conv2d(1, 1, 3)
        self.bn1 = nn.BatchNorm2d(1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(1, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, 3)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 8, 3)
        self.bn6 = nn.BatchNorm2d(8)


        self.fc1 = nn.Linear(8 * 5 * 5, 120)
        self.bnfc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bnfc2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, numClass)



    def forward(self, x):
        x=x.view(-1, 289)
        x=self.bn(x)
        x=x.view(-1, 1,17,17)



        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.bn6(self.conv6(x)))


        x = x.view(-1, 8 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x



def Conv3x3(inPlanes,outPlanes,stride=1):
    ''' 3x3 CNN with padding'''
    return nn.Conv2d(inPlanes, outPlanes, kernel_size=3, stride=stride,padding=1, bias=False)


class Res_CNN_Block(nn.Module):
    def __init__(self, inPlanes, outPlanes, stride=1):
        super(Res_CNN_Block,self).__init__()

        self.conv1 = Conv3x3(inPlanes, outPlanes, stride)
        self.bn1 = nn.BatchNorm2d(outPlanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(outPlanes, outPlanes)
        self.bn2 = nn.BatchNorm2d(outPlanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out



class Res_CNN(nn.Module):
    def __init__(self, block=Res_CNN_Block,numBlocks=10, shapeInput=[1,17,17], numClass=10,inPlanes=8,outPlanes=8):
        super(Res_CNN, self).__init__()

        self.shapeInput=shapeInput
        self.shapeInput_1=shapeInput.copy()
        self.shapeInput_1.insert(0,-1)

        self.numInput=prod(self.shapeInput)
        self.numCNN=self.numInput*outPlanes


        self.bn1D = nn.BatchNorm1d(self.numInput)
        self.conv = Conv3x3(inPlanes=1, outPlanes=inPlanes)
        self.bn2D = nn.BatchNorm2d(inPlanes)


        self.layerBlock = self.makeLayer(block, numBlocks, inPlanes,outPlanes)


        self.fc1 = nn.Linear(self.numCNN, 150)
        self.bnfc1 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, numClass)

        self.relu = nn.ReLU(inplace=True)


    def makeLayer(self, block, numBlocks, inPlanes,outPlanes):
        layers = []
        for i in range(numBlocks):
            layers.append(block(inPlanes=inPlanes, outPlanes=outPlanes))
        return nn.Sequential(*layers)

    def forward(self, x):

        x=x.view(-1, self.numInput)
        x = self.bn1D(x)
        # x=self.relu(x)   # 建议去掉，原结果没去掉
        x=x.view(self.shapeInput_1)

        x=self.conv(x)
        x=self.bn2D(x)
        x=self.relu(x)

        # print(1,x.shape)

        x = self.layerBlock(x)

        # print(2,x.shape)

        x = x.view(-1, self.numCNN)

        # print(3,x.shape)

        x = self.fc1(x)
        # print(4,x.shape)
        x=self.bnfc1(x)
        # print(5,x.shape)
        x=self.relu(x)
        # print(6,x.shape)
        x=self.fc2(x)
        # print(7,x.shape)

        return x



class LSTM1D(nn.Module):
    def __init__(self, seqIn, hidden_size, seqOut, num_layers=2):
        super(LSTM1D, self).__init__()

        self.bn1D = nn.BatchNorm1d(seqIn)
        self.lstm = nn.LSTM(seqIn, hidden_size, num_layers)

        self.fc1 = nn.Linear(hidden_size, 40)
        self.bnfc1 = nn.BatchNorm1d(40)
        self.fc2 = nn.Linear(40, seqOut)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        m,n=x.shape
        x = self.bn1D(x)
        x=x.view(m,-1,n)

        x, _ = self.lstm(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.fc1(x)
        x=self.bnfc1(x)
        x=self.relu(x)
        x=self.fc2(x)

        return x






class LSTM2D(nn.Module):
    def __init__(self, seqIn, hidden_size, seqOut, num_layers=2):
        super(LSTM2D, self).__init__()

        self.seqIn=seqIn
        self.seqIn_1=self.seqIn.copy()
        self.seqIn_1.insert(0,-1)

        self.numInput=prod(self.seqIn)

        self.bn1D = nn.BatchNorm1d(self.numInput)
        self.lstm = nn.LSTM(self.seqIn[-1], hidden_size, num_layers)

        self.fc1 = nn.Linear(self.seqIn[-1]*hidden_size, 60)
        self.bnfc1 = nn.BatchNorm1d(60)
        self.fc2 = nn.Linear(60, seqOut)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x=x.view(-1, self.numInput)
        x = self.bn1D(x)
        x=x.view(self.seqIn_1)
        # print(1,x.shape)

        x, _ = self.lstm(x) # (seq, batch, hidden)

        # print(2,x.shape)
        s, b, h = x.shape
        x = x.view(s,b*h)
        # print(3,x.shape)
        x = self.fc1(x)
        # print(4,x.shape)
        x=self.bnfc1(x)
        # print(5,x.shape)
        x=self.relu(x)
        x=self.fc2(x)
        # print(6,x.shape)

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
