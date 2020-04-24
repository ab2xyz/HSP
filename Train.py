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
    def __init__(self,NN,setTrain,setTest,setValid=None,xShape=(17,17), batchSize=32,numProcessor=6,criterion = nn.CrossEntropyLoss(),optimizer = optim.Adam,cuda=True):
        self.NN=NN
        self.setTrain=setTrain
        self.setTest=setTest
        self.setValid=setValid
        self.criterion=criterion
        self.optimizer=optimizer
        self.opti=self.optimizer(self.NN.parameters())

        self.cuda=cuda
        if self.cuda:
            self.NN.cuda()

        self.batchSize=batchSize
        self.numProcessor=numProcessor

        self.xShape=xShape



    def Train(self,numEpoch,pathSaveNN='../NN.state_dict'):

        lossRecTrain=[]
        lossRecTest=[]
        for epoch in tqdm(range(numEpoch)):

            loaderTrain = torch.utils.data.DataLoader(self.setTrain, batch_size=batchSize,  shuffle=True, num_workers=self.numProcessor)

            self.setTrain.ReadSet()

            self.NN.train()
            lossTrain=0.
            for i, data in enumerate(loaderTrain,0):
                inputs, labels = data

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

                if i%200==199:
                    print(lossTrain/200.)
                    lossTrain=0.

                # print(lossTrain/(i+1))
            lossRecTrain.append(lossTrain/(i+1))

            self.setTrain.ReadGet(numItemKept=1e8)



            ##
            self.NN.eval()


            loaderTest = torch.utils.data.DataLoader(self.setTest, batch_size=batchSize,  shuffle=False, num_workers=self.numProcessor)

            self.setTest.ReadSet()

            lossTest=0.
            for i, data in enumerate(loaderTest,0):
                inputs, labels = data

                inputs=inputs.float()
                labels=labels.long()

                if self.cuda:
                    inputs=inputs.cuda()
                    labels=labels.cuda()

                outputs = self.NN(inputs)

                loss = self.criterion(outputs, labels)

                lossTest+=loss.item()

                # print(lossTest/(i+1))


            lossRecTest.append(lossTest/(i+1))

            self.setTest.ReadGet(numItemKept=0)


            torch.save(self.NN.state_dict(), pathSaveNN)

            plt.figure('loss')
            plt.clf()
            plt.plot(np.log(lossRecTrain),'g',label='train')
            plt.plot(np.log(lossRecTest),'r',label='test')
            plt.grid()
            plt.legend(loc='best')
            plt.pause(0.01)




if __name__=='__main__':


    from Data import Data
    from DataSet import DataSet

    homeCSV='/home/i/iWork/data/csv'
    numProcess=10

    oData=Data(homeCSV,channels=None,ratioSetTrain=0.7,ratioSetTest=0.3,ratioSetValid=0.)

    listCSV4Test=oData.GetListCSV4Test()
    listCSV4Valid=oData.GetListCSV4Valid()
    listCSV4Train=oData.GetListCSV4Train()


    labels=oData.GetLabels()
    numClasses=oData.GetNumClasses()
    branch4Train=oData.GetBranch4Train()
    resize=(1,17,17)

    setTrain=DataSet(homeCSV,listCSV=listCSV4Train,labels=labels,numClasses=numClasses,branch4Train=branch4Train,resize=resize,numProcess=numProcess)
    setTest=DataSet(homeCSV,listCSV=listCSV4Test,labels=labels,numClasses=numClasses,branch4Train=branch4Train,resize=resize,numProcess=numProcess)


    numEpoch=100
    from NN import NN
    Net=NN()
    setValid=None
    xShape=(17,17)
    batchSize=1024*4
    numProcessor=6
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam
    cuda=True

    oTrain=Train(NN=Net,
                 setTrain=setTrain,
                 setTest=setTest,
                 setValid=setValid,
                 xShape=xShape,
                 batchSize=batchSize,
                 numProcessor=numProcessor,
                 criterion = criterion,
                 optimizer = optimizer,
                 cuda=cuda)

    import datetime
    pathSaveNN='/home/i/IGSI/data/res/NN_%s.state_dict'%(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    oTrain.Train(numEpoch=numEpoch,pathSaveNN=pathSaveNN)






'''

class Train():
    def __init__(self,homeCSV,channels=None, ratioSetTrain=0.7,ratioSetTest=0.3,ratioSetValid=0, readMethod=0,numProcess=6,batchSize=32,cuda=True):
        self.homeCSV=homeCSV
        self.channels=channels
        self.ratioSetTrain=ratioSetTrain
        self.ratioSetTest=ratioSetTest
        self.ratioSetValid=ratioSetValid
        self.readMethod=readMethod
        self.numProcess=numProcess
        self.batchSize=batchSize
        self.cuda=cuda



        # self.trainSet=DataSet(homeCSV=self.homeCSV,channels=self.channels,setTrain=1,setTest=0,setValid=0, ratioSetTrain=self.ratioSetTrain,ratioSetTest=self.ratioSetTest,ratioSetValid=self.ratioSetValid,readMethod=self.readMethod)
        # self.testSet=DataSet(homeCSV=self.homeCSV,channels=self.channels,setTrain=0,setTest=1,setValid=0,ratioSetTrain=self.ratioSetTrain,ratioSetTest=self.ratioSetTest,ratioSetValid=self.ratioSetValid,readMethod=self.readMethod)
        #
        # self.trainLoader=torch.utils.data.DataLoader(self.trainSet, batch_size=self.batchSize,shuffle=True, num_workers=self.numProcess)
        # self.testLoader=torch.utils.data.DataLoader(self.testSet, batch_size=self.batchSize,shuffle=False, num_workers=self.numProcess)


        self.trainSet=DataSet_BK(homeCSV=self.homeCSV,channels=self.channels,setTrain=1,setTest=0,setValid=0, ratioSetTrain=self.ratioSetTrain,ratioSetTest=self.ratioSetTest,ratioSetValid=self.ratioSetValid,readMethod=self.readMethod)
        self.testSet=DataSet_BK(homeCSV=self.homeCSV,channels=self.channels,setTrain=0,setTest=1,setValid=0,ratioSetTrain=self.ratioSetTrain,ratioSetTest=self.ratioSetTest,ratioSetValid=self.ratioSetValid,readMethod=self.readMethod)

        self.trainLoader=torch.utils.data.DataLoader(self.trainSet, batch_size=self.batchSize,shuffle=True, num_workers=self.numProcess)
        self.testLoader=torch.utils.data.DataLoader(self.testSet, batch_size=self.batchSize,shuffle=False, num_workers=self.numProcess)


        # self.iNet=Net(numClasses=self.trainSet.numClasses)
        self.iNet=Net(numClasses=34)
        if self.cuda:
            self.iNet.cuda()


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.iNet.parameters())


    def Train(self,numEpoch):
        for epoch in range(-1,numEpoch):  # loop over the dataset multiple times

            ##
            pass  # 开辟进程读取数据






            ##




            ##
            pass  # 开辟进程装载数据






            if epoch>0:
                self.trainSet=DataSet_BK(homeCSV=self.homeCSV,channels=self.channels,setTrain=1,setTest=0,setValid=0, ratioSetTrain=self.ratioSetTrain,ratioSetTest=self.ratioSetTest,ratioSetValid=self.ratioSetValid,readMethod=self.readMethod)

                self.trainLoader=torch.utils.data.DataLoader(self.trainSet, batch_size=self.batchSize,shuffle=True, num_workers=self.numProcess)
            else:
                lossLog=self.homeCSV+'lossLog'
                with open(lossLog,'w') as f:
                    f.close()




            running_loss = 0.0
            for i, data in enumerate(self.trainLoader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs=inputs .cuda()
                labels=labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.iNet(inputs)

                # print(outputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    iRunning_loss=running_loss/20
                    print('[%d, %5d] loss: %.8f' %
                          (epoch + 1, i + 1, iRunning_loss))

                    with open(lossLog,'a') as f:
                        f.writelines('%.8f\n' % ( iRunning_loss) )

                    running_loss = 0.0

            torch.save(self.iNet, self.homeCSV+'iNet')
            torch.save(self.iNet.state_dict(), self.homeCSV+'iNet_dict')

        print('Finished Training')
'''


if __name__=='__main__':

    '''
    from Data import Data
    from DataSet import DataSet

    homeCSV='/home/i/iWork/data/csv'
    numProcess=10

    oData=Data(homeCSV,channels=None,ratioSetTrain=0.7,ratioSetTest=0.3,ratioSetValid=0.)

    listCSV4Test=oData.GetListCSV4Test()
    listCSV4Valid=oData.GetListCSV4Valid()
    listCSV4Train=oData.GetListCSV4Train()


    labels=oData.GetLabels()
    numClasses=oData.GetNumClasses()
    branch4Train=oData.GetBranch4Train()

    setTrain=DataSet(homeCSV,listCSV=listCSV4Train,labels=labels,numClasses=numClasses,branch4Train=branch4Train,numProcess=numProcess)
    setTest=DataSet(homeCSV,listCSV=listCSV4Test,labels=labels,numClasses=numClasses,branch4Train=branch4Train,numProcess=numProcess)

    loaderTrain = torch.utils.data.DataLoader(setTrain, batch_size=32,
                                          shuffle=True, num_workers=10)

    loaderTest = torch.utils.data.DataLoader(setTest, batch_size=32,
                                          shuffle=True, num_workers=10)

    dataiter = iter(loaderTrain)
    iData, iLabel = dataiter.next()
    print(iData.shape,iLabel.shape)
    '''


    # oTrain=Train()



#
