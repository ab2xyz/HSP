#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import uproot
import pandas as pd
import os
from multiprocessing import Process
from tqdm import tqdm
import time

import re

from torch.utils.data import  Dataset
from random import  choice
from multiprocessing import Pool,Process,Queue

from Channel import  Channel
from Data import Data



class DataSet(Dataset,Channel):
    def __init__(self,homeCSV,listCSV,labels,numClasses,branch4Train, resize=(17,17),numProcess=10):
        self.homeCSV=homeCSV
        if self.homeCSV.strip()[-1]!='/':
            self.homeCSV=self.homeCSV+'/'

        self.listCSV=listCSV
        self.labels=labels
        self.numClasses=numClasses
        self.branch4Train=branch4Train
        self.branch4Train.sort()

        self.numProcess=numProcess

        self.resize=resize


        self.branch4Train.sort()

        self.data=pd.DataFrame(columns=self.branch4Train)
        self.label=np.array([])

        self.prob=dict(zip(range(self.numClasses),np.ones(self.numClasses)))
        self.dictNumEventChannel=dict(zip(range(self.numClasses),np.zeros(self.numClasses)))

        self.counterRead=0
        self.probPositive=self.numClasses

        self.Label2Data()

        self.branchSel={}

        self.ReadCSV()



    def Label2Data(self):
        label2Channel={}
        for channle, label in self.labels.items():
            if label in label2Channel.keys():
                label2Channel[label].append(channle)
                label2Channel[label].sort()
            else:
                label2Channel[label]=[channle]

        self.label2Channel=label2Channel

        label2CSV={}
        for channel, label in self.labels.items():
            if label in label2CSV.keys():
                label2CSV[label].extend([self.homeCSV+channel+'/'+x for x in self.listCSV[channel]])
                label2CSV[label].sort()
            else:
                label2CSV[label]=[self.homeCSV+channel+'/'+x for x in self.listCSV[channel]]

        self.label2CSV=label2CSV

        logLabel2CSV=self.homeCSV+'label2CSVLog'
        with open(logLabel2CSV,'w') as f:
            for x, y in  self.label2CSV.items():
                f.writelines('label=%d  -- number of csvs=%d \n'%(x,len(y)))
                [f.writelines(z+'\n') for z in y]
                f.writelines('='*80+'\n'*2)


    def Resize(self,x):
        return np.resize(x,self.resize)


    def __len__(self):
        return len(self.data)


    def __getitem__(self,idx):

        iData=self.data[idx,:]
        iData=self.Resize(iData)

        iLabel=self.label[idx]
        iUid=self.uid[idx]


        data=(iData,iLabel,iUid)

        return data


    def ReadCSV(self):

        ## 分开处理csvlist： 读取-处理
        pool = Pool(self.numProcess,maxtasksperchild=1)

        data_label=pool.map(self.ReadCSV_OneFile, range(self.numClasses), chunksize=1)
        dataList=[x[0] for x in data_label]
        labelList=[x[1] for x in data_label]
        uidList=[x[2] for x in data_label]
        classList=[x[3] for x in data_label]

        # self.data=pd.concat((dataList),sort=True)
        self.data=np.concatenate(dataList,axis=0)
        self.label=np.concatenate(labelList,axis=0)
        self.uid=np.concatenate(uidList,axis=0)

        pool.close()

        dataSize=[x.shape[0] for x in dataList]

        for idx in range(self.numClasses):
            self.dictNumEventChannel[classList[idx]]=dataSize[idx]

        self.counterRead+=1


    def ReadCSV_OneFile(self,iClass, q=None, iCSV=None):

        if iCSV is None:
            iCSV=choice(self.label2CSV[iClass])


        data=pd.DataFrame(columns=self.branch4Train)
        label=np.array([])
        uid=pd.DataFrame(columns=['uid'])


        iChannel=iCSV.split('/')[-2]
        if iChannel in self.branchSel:
            iBranchSel=self.branchSel[iChannel]
        else:
            branchAll=pd.read_csv(iCSV,nrows=0).columns.tolist()
            iBranchSel=list(set(branchAll).intersection(set(self.branch4Train)))
            iBranchSel.sort()
            self.branchSel[iChannel]=iBranchSel


        iData=pd.read_csv(iCSV,usecols=iBranchSel)
        iLabel=np.ones((iData.shape[0]),dtype=np.long)*iClass
        iUid=pd.read_csv(iCSV,usecols=['uid'])

        iData=pd.concat((data,iData),sort=True).fillna(0).values
        iLabel=np.r_[label,iLabel]
        iUid=pd.concat((uid,iUid),sort=True).fillna(0).values

        if q is None:
            return (iData,iLabel,iUid, iClass)

        else:
            q.put([iData,iLabel,iUid, iClass])


    def Prob(self):
        numEventChannelAveNext=np.sum(list(self.dictNumEventChannel.values()))/self.probPositive*(self.counterRead+1)

        for key in self.dictNumEventChannel:
            if self.dictNumEventChannel[key]>numEventChannelAveNext:
                self.prob[key]=0
            else:
                self.probPositive+=1
                self.prob[key]=1


        with open(self.homeCSV+'probLog','w') as f:
            [f.writelines('%d :%.1f \n'%(i, self.dictNumEventChannel[i]/numEventChannelAveNext/self.counterRead*(self.counterRead+1))) for i in self.dictNumEventChannel]


    def ReadSet(self):
        self.Prob()

        self.q=Queue()

        # p=[print(iClass) for iClass in range(self.numClasses) if self.prob[iClass]>0]

        p=[Process(target=self.ReadCSV_OneFile, args=(iClass,self.q,)) for iClass in range(self.numClasses) if self.prob[iClass]>0]
        [ip.start() for ip in p]


    def ReadGet(self,numItemKept=100000):
        self.counterRead+=1

        dataList=[]
        labelList=[]
        uidList=[]
        for key in self.prob:
            if self.prob[key]<1:
                continue

            iData, iLabel, iUid,iClass=self.q.get()

            self.dictNumEventChannel[iClass]+=iData.shape[0]

            dataList.append(iData)
            labelList.append(iLabel)
            uidList.append(iUid)



        data=np.concatenate(dataList,axis=0)
        label=np.concatenate(labelList,axis=0)
        uid=np.concatenate(uidList,axis=0)

        if numItemKept<=0:
            self.data=data
            self.label=label
            self.uid=uid

        else:

            if data.shape[0]<numItemKept:
                # data=np.concatenate((self.data,data),axis=0)
                # label=np.concatenate((self.label,label),axis=0)

                data=np.r_[self.data,data]
                label=np.r_[self.label,label]
                uid=np.r_[self.uid,uid]

            self.data=data[-int(numItemKept):,:]
            self.label=label[-int(numItemKept):]
            self.uid=uid[-int(numItemKept):]

    def SetDataLabel(self,data,label,uid):
        self.data=data
        self.label=label
        self.uid=uid


if __name__=='__main__':
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

    setTrain.ReadSet()

    setTrain.ReadGet(numItemKept=1000000)


    import torch
    trainLoader = torch.utils.data.DataLoader(setTrain, batch_size=4,
                                          shuffle=True, num_workers=1)

    dataiter = iter(trainLoader)
    iData, iLabel, iUid = dataiter.next()

    print(iData),print(iLabel),print(iUid)
    print(iData.shape,iLabel.shape,iUid.shape)




#
