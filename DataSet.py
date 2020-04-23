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
from multiprocessing import Pool

from Channel import  Channel
from Data import Data



class DataSet(Dataset,Channel):
    def __init__(self,homeCSV,listCSV,labels,numClasses,branch4Train,numProcess=10):
        self.homeCSV=homeCSV
        if self.homeCSV.strip()[-1]!='/':
            self.homeCSV=self.homeCSV+'/'

        self.listCSV=listCSV
        self.labels=labels
        self.numClasses=numClasses
        self.branch4Train=branch4Train
        self.numProcess=numProcess


        self.branch4Train.sort()

        self.data=pd.DataFrame(columns=self.branch4Train)
        self.label=np.array([])

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


    def __len__(self):
        return len(self.data)


    def __getitem__(self,idx):

        iData=self.data.iloc[idx][self.branch4Train[::]].values

        iLabel=self.label[idx]


        data=(iData,iLabel)

        return data


    def ReadCSV(self):

        ## 分开处理csvlist： 读取-处理
        pool = Pool(self.numProcess,maxtasksperchild=1)

        data_label=pool.map(self.ReadCSV_OneFile, range(self.numClasses), chunksize=1)
        dataList=[x[0] for x in data_label]
        labelList=[x[1] for x in data_label]

        self.data=pd.concat((dataList),sort=True)
        self.label=np.concatenate(labelList,axis=0)

        pool.close()

    def ReadCSV_OneFile(self,iClass):

        data=pd.DataFrame(columns=self.branch4Train)
        label=np.array([])


        iCSV=choice(self.label2CSV[iClass])

        iChannel=iCSV.split('/')[-2]
        if iChannel in self.branchSel:
            iBranchSel=self.branchSel[iChannel]
        else:
            branchAll=pd.read_csv(iCSV,nrows=0).columns.tolist()
            iBranchSel=list(set(branchAll).intersection(set(self.branch4Train)))
            self.branchSel[iChannel]=iBranchSel


        iData=pd.read_csv(iCSV,usecols=iBranchSel)
        iLabel=np.ones((iData.shape[0]),dtype=np.long)*iClass

        iData=pd.concat((data,iData),sort=True).fillna(0)
        iLabel=np.r_[label,iLabel]


        return (iData,iLabel)




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

    setTrain=DataSet(homeCSV,listCSV=listCSV4Train,labels=labels,numClasses=numClasses,branch4Train=branch4Train,numProcess=numProcess)

    import torch
    trainLoader = torch.utils.data.DataLoader(setTrain, batch_size=4,
                                          shuffle=True, num_workers=1)

    dataiter = iter(trainLoader)
    iData, iLabel = dataiter.next()

    print(iData),print(iLabel)
    print(iData.shape,iLabel.shape)


#
