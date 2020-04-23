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
    def __init__(self,homeCSV,channels,setTrain=1,setTest=0,setValid=0,ratioSetTrain=0.7,ratioSetTest=0.3,ratioSetValid=0.,readMethod=0,numProcess=10):
        super(DataSet,self).__init__(homeCSV,channels)

        assert (((int(setTrain==0)+int(setTest==0)+int(setValid==0))==2) and ((int(setTrain==1)+int(setTest==1)+int(setValid==1))==1)),'Ex.: setTrain=1,setTest=0,setValid=0'

        oData=Data(homeCSV=homeCSV,channels=channels,ratioSetTrain=ratioSetTrain,ratioSetTest=ratioSetTest,ratioSetValid=ratioSetValid)

        self.numProcess=numProcess

        if setTrain:
            self.listCSV=oData.GetListCSV4Train()

        if setTest:
            self.listCSV=oData.GetListCSV4Test()

        if setValid:
            self.listCSV=oData.GetListCSV4Valid()

        self.labels=oData.GetLabels()

        self.numClasses=oData.GetNumClasses()
        self.branch4Train= oData.GetBranch4Train()




        self.branch4Train.sort()

        self.data=pd.DataFrame(columns=self.branch4Train)
        self.label=np.array([])

        self.readMethod=readMethod


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
        # iData.resize((1,17,17))

        iLabel=self.label[idx]

        # iData=torch.from_numpy(iData).float()
        # iLabel=torch.from_numpy(np.array(iLabel)).long()  #.astype(np.long)

        print(iData.shape,iLabel.shape)
        print(type(iData),type(iLabel))

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
    homeCSV =   '/home/i/iWork/data/csv'
    channels  =None
    setTrain=1
    setTest=0
    setValid=0
    ratioSetTrain=0.7
    ratioSetTest=0.3
    ratioSetValid=0.
    readMethod=0
    numProcess=10
    oDataSet=DataSet(homeCSV=homeCSV,
                     channels=channels,
                     setTrain=setTrain,
                     setTest=setTest,
                     setValid=setValid,
                     ratioSetTrain=ratioSetTrain,
                     ratioSetTest=ratioSetTest,
                     ratioSetValid=ratioSetValid,
                     readMethod=readMethod,
                     numProcess=numProcess)


# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)


#
