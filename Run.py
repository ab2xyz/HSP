#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from RootSetSplit import RootSetSplit
from Root2CSV import Root2CSV
from Channel2Label import Channel2Label
from Branch4Train import Branch4Train

## Step 1.

homeRoot='/home/i/iWork/data/root'
homeData='/home/i/IGSI/data/data'
channels=None
train=0.7

oRootSetSplit=RootSetSplit(homeRoot=homeRoot, homeData=homeData, channels=channels,train=train)
homeCSVTrain,homeCSVTest=oRootSetSplit.GetHomeCSV()

## Step 2.
## Channel

## Step3.
## Root2CSV

numEventPerFile=1024
numProcess=10


oRoot2CSVTrain=Root2CSV(homeRoot=homeRoot,
        homeCSV=homeCSVTrain,
        channels=channels,
        numEventPerFile=numEventPerFile,
        numProcess=numProcess)


oRoot2CSVTrain.ReRun()
checkRoot2CSVTrain=oRoot2CSVTrain.Check2Finish(timeSecondDelta=10,timeSecondTotal=7200,label='oRoot2CSVTrain')



oRoot2CSVTest=Root2CSV(homeRoot=homeRoot,
        homeCSV=homeCSVTest,
        channels=channels,
        numEventPerFile=numEventPerFile,
        numProcess=numProcess)


oRoot2CSVTest.ReRun()
checkRoot2CSVTest=oRoot2CSVTest.Check2Finish(timeSecondDelta=1,timeSecondTotal=7200,label='oRoot2CSVTest')



## Step 4
## Channel2Label

oChannel2Label=Channel2Label(homeCSV=homeCSVTrain,channels=channels)
channel2Label=oChannel2Label.GetChannel2Label()
numClass=oChannel2Label.GetNumClass()


## Step 5
## Branch
oBranch4Train=Branch4Train(homeCSV=homeCSVTrain,channels=channels)
branch4Train=oBranch4Train.Branch4Train()


## Step 6
## DataSet
# setTrain=DataSet(homeCSV=homeCSVTrain,
#                  channels=channels,
#                  channel2Label=channel2Label,
#                  numClasses=numClasses,
#                  branch4Train=branch4Train,
#                  resize=resize,
#                  numProcess=numProcess)



## Step 7
## Train

## Step 8
## Test




print('END')


#
