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
import os



class Train():
    def __init__(self,
                 NN,
                 setTrain,
                 setTest,
                 batchSize=1024,
                 numProcess=6,
                 cuda=True,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer = optim.Adam):

        self.NN=NN
        self.setTrain=setTrain
        self.setTest=setTest

        self.batchSize=batchSize
        self.numProcess=numProcess

        self.criterion=criterion
        self.optimizer=optimizer
        self.opti=self.optimizer(self.NN.parameters(),weight_decay=1e-4)

        self.cuda=cuda
        if self.cuda:
            self.NN.cuda()



    def Train(self,numEpoch=10,homeRes='',codeSave='',numItemKeep=5e6):

        homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'
        homeRes=homeRes+codeSave+'/'
        os.makedirs(homeRes,exist_ok=True)

        self.setTrain.ReadTrainSet()
        self.setTest.ReadTrainSet()
        lossRecTrain=[]
        lossRecTest=[]
        for epoch in tqdm(range(numEpoch)):

            self.setTrain.ReadTrainGet(numItemKeep=numItemKeep)

            loaderTrain = torch.utils.data.DataLoader(self.setTrain, batch_size=self.batchSize,  shuffle=True, num_workers=self.numProcess)

            if epoch<numEpoch-1:
                self.setTrain.ReadTrainSet()

            self.NN.train()
            lossTrain=0.
            for i, data in enumerate(loaderTrain,0):
                inputs, labels, uids= data

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


            ##
            self.NN.eval()

            self.setTest.ReadTrainGet(numItemKeep=0)

            loaderTest = torch.utils.data.DataLoader(self.setTest, batch_size=self.batchSize,  shuffle=False, num_workers=self.numProcess)

            if epoch<numEpoch-1:
                self.setTest.ReadTrainSet()

            lossTest=0.
            for i, data in enumerate(loaderTest,0):
                inputs, labels,uids = data
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






            pathSaveNN=homeRes+'NN'+codeSave+'.plt'
            torch.save(self.NN.state_dict(), pathSaveNN)
            pathSaveResize=homeRes+'Resize'+codeSave+'.dat'
            with open(pathSaveResize,'w') as f:
                [f.writelines('%d '%x) for x in self.setTrain.resize]

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



    plt.show()


#
