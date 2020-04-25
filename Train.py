#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import  tqdm
import matplotlib.pyplot as plt




class Train():
    def __init__(self,NN,setTrain,setTest,setValid=None, batchSize=32,numProcessor=6,criterion = nn.CrossEntropyLoss(),optimizer = optim.Adam,cuda=True):
        self.NN=NN
        self.setTrain=setTrain
        self.setTest=setTest
        self.setValid=setValid
        self.criterion=criterion
        self.optimizer=optimizer
        self.opti=self.optimizer(self.NN.parameters(),weight_decay=1e-4)

        self.cuda=cuda
        if self.cuda:
            self.NN.cuda()

        self.batchSize=batchSize
        self.numProcessor=numProcessor





    def Train(self,numEpoch,homeRes='',codeSave='',numItemKept=5e6):

        lossRecTrain=[]
        lossRecTest=[]
        for epoch in tqdm(range(numEpoch)):

            loaderTrain = torch.utils.data.DataLoader(self.setTrain, batch_size=batchSize,  shuffle=True, num_workers=self.numProcessor)

            self.setTrain.ReadSet()

            self.NN.train()
            lossTrain=0.
            for i, data in enumerate(loaderTrain,0):
                inputs, labels = data

                if inputs.shape[0]==1:
                    continue

                inputs=inputs.float()
                labels=labels.long()

                if self.cuda:
                    inputs=inputs.cuda()
                    labels=labels.cuda()

                self.opti.zero_grad()
                outputs = self.NN(inputs)


                loss = self.criterion(outputs, labels)
                loss.backward()
                self.opti.step()

                lossTrain+=loss.item()


            lossRecTrain.append(lossTrain/(i+1))

            self.setTrain.ReadGet(numItemKept=numItemKept)



            ##
            self.NN.eval()


            loaderTest = torch.utils.data.DataLoader(self.setTest, batch_size=batchSize,  shuffle=False, num_workers=self.numProcessor)

            self.setTest.ReadSet()

            lossTest=0.
            for i, data in enumerate(loaderTest,0):
                inputs, labels = data
                if inputs.shape[0]==1:
                    continue

                inputs=inputs.float()
                labels=labels.long()

                if self.cuda:
                    inputs=inputs.cuda()
                    labels=labels.cuda()

                outputs = self.NN(inputs)

                loss = self.criterion(outputs, labels)

                lossTest+=loss.item()

                if i==0:
                    softmax=nn.Softmax(dim=1)
                    print('\n',softmax(outputs).mean(dim=0))



            lossRecTest.append(lossTest/(i+1))

            self.setTest.ReadGet(numItemKept=0)


            homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'
            pathSaveNN=homeRes+'NN'+codeSave+'.plt'
            torch.save(self.NN.state_dict(), pathSaveNN)

            plt.figure('loss',figsize=(14,8))
            plt.clf()
            plt.subplot(121)
            plt.plot(np.log(lossRecTrain),'g',label='train')
            plt.plot(np.log(lossRecTest),'r',label='test')
            plt.grid()
            plt.legend(loc='best')

            plt.subplot(122)
            plt.plot(lossRecTrain,'g',label='train')
            plt.plot(lossRecTest,'r',label='test')
            plt.grid()
            plt.legend(loc='best')

            plt.pause(0.02)




if __name__=='__main__':


    from Data import Data
    from DataSet import DataSet

    homeCSV='/home/i/iWork/data/csv'
    homeRes='/home/i/iWork/data/res'
    numProcess=10
    channels='45'
    channels=['T06_T06_45_Dch','T06_DPM_45']

    oData=Data(homeCSV,channels=channels,ratioSetTrain=0.7,ratioSetTest=0.3,ratioSetValid=0.)

    codeSave=oData.Write(homeRes)
    print(codeSave)

    listCSV4Test=oData.GetListCSV4Test()
    listCSV4Valid=oData.GetListCSV4Valid()
    listCSV4Train=oData.GetListCSV4Train()


    labels=oData.GetLabels()
    numClasses=oData.GetNumClasses()
    print(numClasses)
    branch4Train=oData.GetBranch4Train()
    resize=(240)


    setTrain=DataSet(homeCSV,listCSV=listCSV4Train,labels=labels,numClasses=numClasses,branch4Train=branch4Train,resize=resize,numProcess=numProcess)
    setTest=DataSet(homeCSV,listCSV=listCSV4Test,labels=labels,numClasses=numClasses,branch4Train=branch4Train,resize=resize,numProcess=numProcess)


    numEpoch=300
    from DNN import NN
    Net=NN(numClasses)
    setValid=None
    batchSize=1024*4
    numProcessor=6
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam
    cuda=True

    oTrain=Train(NN=Net,
                 setTrain=setTrain,
                 setTest=setTest,
                 setValid=setValid,
                 batchSize=batchSize,
                 numProcessor=numProcessor,
                 criterion = criterion,
                 optimizer = optimizer,
                 cuda=cuda)

    numItemKept=5e6

    oTrain.Train(numEpoch=numEpoch,homeRes=homeRes,codeSave=codeSave,numItemKept=numItemKept)


    plt.show()


#
