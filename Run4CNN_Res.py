#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from Root2Channel import Root2Channel
from RootSetSplit import RootSetSplit
from Root2CSV import Root2CSV
from Channel2Label import Channel2Label
from Branch4Train import Branch4Train
from DataSet import DataSet
from NN import DNN_BN, DNN_Dropout, DNN_Res_Manual,Res_DNN_Block,Res_DNN, CNN, CNN_BN,Res_CNN_Block,Res_CNN, LSTM1D, LSTM2D
from Train import  Train
from Test import Test
from Post import WriteEffi2CSV



## Step 0.
## Root2Channel
## Get Train/Test  and CSV/Root
homeRoot='/home/i/IGSI/data/root'
oRoot2Channel=Root2Channel(homeRoot=homeRoot)

## Step 1.
## RootSetSplit
## Get Train/Test  and CSV/Root

homeData='/home/i/IGSI/data/data'
channels='55'
# channels=['T06_DPM_45','T06_T06_45_Dch']
train=0.8

oRootSetSplit=RootSetSplit(homeRoot=homeRoot, homeData=homeData, channels=channels,train=train)
homeCSVTrain,homeCSVTest=oRootSetSplit.GetHomeCSV()
homeRootTrain,homeRootTest=oRootSetSplit.GetHomeRoot()

## Step 2.
## Channel
## No Run

## Step3.
## Root2CSV
## Double Run :  Train / Test

numEventPerFile=1024
numProcess=4


oRoot2CSVTrain=Root2CSV(homeRoot=homeRootTrain,
        homeCSV=homeCSVTrain,
        channels=channels,
        numEventPerFile=numEventPerFile,
        numProcess=numProcess)


oRoot2CSVTrain.ReRun()
checkRoot2CSVTrain=oRoot2CSVTrain.Check2Finish(timeSecondDelta=10,timeSecondTotal=7200,label='oRoot2CSVTrain')



oRoot2CSVTest=Root2CSV(homeRoot=homeRootTest,
        homeCSV=homeCSVTest,
        channels=channels,
        numEventPerFile=numEventPerFile,
        numProcess=numProcess)


oRoot2CSVTest.ReRun()
checkRoot2CSVTest=oRoot2CSVTest.Check2Finish(timeSecondDelta=10,timeSecondTotal=7200,label='oRoot2CSVTest')



## Step 4
## Channel2Label
## Single Run :  homeCSVTrain

oChannel2Label=Channel2Label(homeCSV=homeCSVTrain,channels=channels)
channel2Label=oChannel2Label.GetChannel2Label()
numClass=oChannel2Label.GetNumClass()


## Step 5
## Branch
## Single Run  :homeCSVTrain
oBranch4Train=Branch4Train(homeCSV=homeCSVTrain,channels=channels)
branch4Train=oBranch4Train.Branch4Train()


## Step 6
## DataSet
## Double Run : Train / Test
numItemKeep=5e5
# resize=[1,17,17]
resize=[232]
# resize=[17,17]


setTrain=DataSet(homeCSV=homeCSVTrain,
                 channels=channels,
                 channel2Label=channel2Label,
                 numClass=numClass,
                 branch4Train=branch4Train,
                 resize=resize)

# setTrain.ReadTrainSet()
# setTrain.ReadTrainGet(numItemKeep=numItemKeep)




setTest=DataSet(homeCSV=homeCSVTest,
                 channels=channels,
                 channel2Label=channel2Label,
                 numClass=numClass,
                 branch4Train=branch4Train,
                 resize=resize)


# # setTest.ReadTestSet(channels=['T08_DPM_45'],numCSV=2)
# # setTest.ReadTestGet()
# setTest.ReadTest(channels=['T08_DPM_45'],numCSV=0,numProcess=numProcess)


## Step 7
## NN
# oDNN=DNN(resize[0],numClass)
#
# oDNNTestBN=DNNTestBN(resize[0],numClass)
#
# oResidualDNN=ResidualDNN(block=ResidualDNNBlock,numBlocks=10, numInput=resize[0], numClass=numClass)
#
# oResDNN=ResDNN(resize[0],numClass)


#DNN_BN, DNN_Dropout, DNN_Res_Manual,Res_DNN_Block,
oRes_DNN=Res_DNN(block=Res_DNN_Block,numBlocks=10, numInput=resize[0], numClass=numClass)
oCNN=CNN(resize,numClass)
oCNN_BN=CNN_BN(resize,numClass)
oRes_CNN=Res_CNN(block=Res_CNN_Block,numBlocks=10, shapeInput=resize, numClass=numClass,inPlanes=8,outPlanes=8)
oRes_LSTM1D=LSTM1D(seqIn=resize[0], hidden_size=80, seqOut=numClass, num_layers=2)
# oRes_LSTM2D=LSTM2D(seqIn=resize, hidden_size=16, seqOut=numClass, num_layers=2)
# oRes_LSTM1D_8=LSTM1D(seqIn=resize[0], hidden_size=80, seqOut=numClass, num_layers=8)
# oRes_LSTM1D_4=LSTM1D(seqIn=resize[0], hidden_size=80, seqOut=numClass, num_layers=4)

oNN=oRes_LSTM1D


## Step 7
## Train
## Single Run : Train

batchSize=512*4
cuda=True

numEpoch=10000
homeRes='/home/i/IGSI/data/res'
codeSave='LSTM1D_55'

numItemKeep=5e6


oTrain=Train(NN=oNN,
                 setTrain=setTrain,
                 setTest=setTest,
                 batchSize=batchSize,
                 numProcess=numProcess,
                 cuda=cuda)

oTrain.Train(numEpoch=numEpoch,homeRes=homeRes,codeSave=codeSave,numItemKeep=numItemKeep)




## Step 8
## Test
numCSVReadPerChannel=0
effiBkgTarget=0.999

setTest=dict(zip(list(channel2Label.keys()),[DataSet(homeCSV=homeCSVTest,
                 channels=channels,
                 channel2Label=channel2Label,
                 numClass=numClass,
                 branch4Train=branch4Train,
                 resize=resize) for i in range(len(channel2Label))]))

oTest=Test(homeCSV=homeCSVTest,
           homeRes=homeRes,
            NN=oNN,
            setTest=setTest,
            codeSave=codeSave,
            batchSize=batchSize,
            numProcess=numProcess,
            cuda=cuda)

oTest.ReadCSV(numCSVReadPerChannel=numCSVReadPerChannel)
oTest.RunNN()

oTest.GetCuts(effiBkgTarget=effiBkgTarget)
oTest.GetTable()

oTest.GetDPMPerformance()



## Step 9
## Post
WriteEffi2CSV(homeRes=homeRes,codeSave=codeSave)




plt.show()
print('END')


#
