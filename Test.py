#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import  tqdm
import matplotlib.pyplot as plt
import json
from DataSet import DataSet




class Test():
    def __init__(self,homeCSV,homeRes,NN,codeSave,numFilesCut=20, batchSize=1024, numProcessor=1, cuda=True):

        self.homeCSV=homeCSV if homeCSV.strip()[-1]=='/' else homeCSV+'/'
        self.homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'

        self.codeSave=codeSave

        self.numFilesCut=int(numFilesCut)

        self.batchSize=batchSize
        self.numProcessor=numProcessor
        self.cuda=cuda


        jsonName=self.homeRes+'data'+self.codeSave+'.json'
        with open(jsonName,'r') as f:
            dataDict=json.load(f)

        self.listCSV4Test=dataDict['listCSV4Test']
        self.listCSV4Valid=dataDict['listCSV4Valid']
        self.listCSV4Train=dataDict['listCSV4Train']
        self.labels=dataDict['labels']
        self.numClasses=dataDict['numClasses']
        print(self.numClasses)
        self.branch4Train=dataDict['branch4Train']

        self.branch4Train.sort()

        self.NN=NN(self.numClasses)

        nnDictName=self.homeRes+'NN'+self.codeSave+'.plt'
        self.NN.load_state_dict(torch.load(nnDictName))

        if self.cuda:
            self.NN=self.NN.cuda()

        resizeName=self.homeRes+'Resize'+codeSave+'.dat'
        self.resize=np.loadtxt(resizeName).astype(int)


        self.NN.eval()

        self.softmax=torch.nn.Softmax(dim=1)


        self.setTest={}
        self.loaderTest={}



        self.ReadCSV()







    def ReadCSV(self):

        for iChannel in tqdm(self.labels):
            self.setTest[iChannel]=DataSet(homeCSV=self.homeCSV,listCSV=self.listCSV4Test,labels=self.labels,numClasses=self.numClasses,branch4Train=self.branch4Train,resize=self.resize,numProcess=self.numProcessor)

            data=None
            label=None

            counterCSV=0
            for iCSV in tqdm(self.listCSV4Test[iChannel]):
                counterCSV+=1
                if (self.numFilesCut>0) and (counterCSV>self.numFilesCut):
                    continue

                iReadCSV=self.homeCSV+iChannel+'/'+iCSV
                iReadClass=self.labels[iChannel]
                iData,iLabel,iClass=self.setTest[iChannel].ReadCSV_OneFile(iClass=iReadClass, iCSV=iReadCSV)

                if data is None:
                    data=iData
                    label=iLabel
                else:
                    data=np.r_[data,iData]
                    label=np.r_[label,iLabel]


            self.setTest[iChannel].SetDataLabel(data=data,label=label)
            self.loaderTest[iChannel]=torch.utils.data.DataLoader(self.setTest[iChannel], batch_size=self.batchSize,  shuffle=False, num_workers=self.numProcessor)








    def Test(self):

        for iChannel in tqdm(self.labels):
            loader=self.loaderTest[iChannel]

            lossTest=0.
            for i, data in enumerate(loader,0):
                inputs, labels = data

                if inputs.shape[0]==1:
                    continue

                inputs=inputs.float()
                labels=labels.long()

                if self.cuda:
                    inputs=inputs.cuda()
                    labels=labels.cuda()

                outputs = self.NN(inputs)


                if i==0:
                    outputsMean=self.softmax(outputs).mean(dim=0)
                    print('\n',labels[0],outputsMean,iChannel)







if __name__=='__main__':
    import time

    homeCSV='/home/i/iWork/data/csv'
    homeRes='/home/i/iWork/data/res'
    from DNN import NN
    codeSave='_20200425_185354'

    numFilesCut=2
    batchSize=1024
    numProcessor=1
    cuda=True


    oTest=Test(homeCSV=homeCSV,homeRes=homeRes,NN=NN,codeSave=codeSave,numFilesCut=numFilesCut, batchSize=batchSize, numProcessor=numProcessor, cuda=cuda)

    for iTime in range(10000):
        oTest.Test()
        print('\n'+'-'*80+'\n')
        time.sleep(200)








#
