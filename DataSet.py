#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This version works fine with Queue and starmap!

'''


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
    def __init__(self,homeCSV,channels,channel2Label,numClass,branch4Train, resize=(17,17)):
        super(DataSet,self).__init__(homeCSV,channels)

        self.channel2Label=channel2Label
        self.numClass=numClass
        self.branch4Train=branch4Train
        self.branch4Train.sort()

        self.resize=resize
        self.resizeProd=np.prod(self.resize)


        self.Label2Channel()
        self.ProbInit()
        self.channel_csv={}
        self.branchSel={}




    def Label2Channel(self):
        self.label2Channel={}
        for iChannel in self.channel2Label:
            iLabel=self.channel2Label[iChannel]
            if not iLabel in self.label2Channel:
                self.label2Channel[iLabel]=[]
            self.label2Channel[iLabel].append(iChannel)

        # self.PrintDict(self.label2Channel)



    def ProbInit(self):

        self.prob=dict(zip(range(self.numClass),np.ones(self.numClass)))
        self.dictLabelNumItem=dict(zip(range(self.numClass),np.zeros(self.numClass)))

        self.counterRead=0
        self.probPositive=self.numClass



    def Prob(self):

        numEventMeanLabelNext=np.sum(list(self.dictLabelNumItem.values()))/self.probPositive*(self.counterRead+1)

        for key in self.dictLabelNumItem:
            if self.dictLabelNumItem[key]>numEventMeanLabelNext:
                self.prob[key]=0
            else:
                self.probPositive+=1
                self.prob[key]=1

        if numEventMeanLabelNext>0:
            with open(self.homeCSV+'probLog','w') as f:
                [f.writelines('%d :%.1f \n'%(i, self.dictLabelNumItem[i]/numEventMeanLabelNext/self.counterRead*(self.counterRead+1))) for i in self.dictLabelNumItem]



    def Label2CSV(self,iLabel):
        iChannel=np.random.choice(self.label2Channel[iLabel])

        if not iChannel in self.channel_csv:
            self.channel_csv[iChannel]=[x for x in os.listdir(self.homeCSV+iChannel) if x[-4:]=='.csv']

        iCSV=self.homeCSV+iChannel+'/'+np.random.choice(self.channel_csv[iChannel])

        return iCSV






    def Label_CSV2Read4Train(self):
        self.Prob()

        label4Read=[iClass for iClass in range(self.numClass) if self.prob[iClass]>0]

        self.label_csv={}
        for iLabel in label4Read:
            iCSV=self.Label2CSV(iLabel)
            self.label_csv[iLabel]=iCSV




    def ReadCSVOneFile(self,iLabel, iCSV, q=None):

        iChannel=iCSV.split('/')[-2]
        if iChannel in self.branchSel:
            iBranchSel=self.branchSel[iChannel]
        else:
            branchAll=pd.read_csv(iCSV,nrows=0).columns.tolist()
            iBranchSel=list(set(branchAll).intersection(set(self.branch4Train)))
            iBranchSel.sort()
            self.branchSel[iChannel]=iBranchSel

        data=pd.DataFrame(columns=self.branch4Train)
        iData=pd.read_csv(iCSV,usecols=iBranchSel)
        data=pd.concat((data,iData),sort=True).fillna(0).values


        mData,nData=data.shape
        assert(self.resizeProd>=nData),"'resize' is less than the real data, and data will be cut... Please enlarge the resize."
        if self.resizeProd>nData:
            zeros=np.zeros((mData,self.resizeProd-nData))
            data=np.c_[data,zeros]

        reshape=[mData]
        for i in self.resize:
            reshape.append(i)
        data=data.reshape(reshape)


        uid=pd.read_csv(iCSV,usecols=['uid']).values[:,0]

        label=np.ones((uid.shape[0]))*iLabel

        if q is None:
            return (data,label,uid, iLabel)
        else:
            q.put([data,label,uid, iLabel])




    def ReadTrainSet(self):
        self.Label_CSV2Read4Train()
        self.q=Queue()

        p=[Process(target=self.ReadCSVOneFile, args=(iLabel,iCSV, self.q,)) for iLabel,iCSV in self.label_csv.items()]
        [ip.start() for ip in p]



    def ReadTrainGet(self,numItemKeep=1e6):

        self.counterRead+=1

        dataList=[]
        labelList=[]
        uidList=[]

        for key in self.prob:
            if self.prob[key]<1:
                continue


            iData, iLabel, iUid,iClass=self.q.get()

            self.dictLabelNumItem[iClass]+=iData.shape[0]

            dataList.append(iData)
            labelList.append(iLabel)
            uidList.append(iUid)



        data=np.concatenate(dataList,axis=0)
        label=np.concatenate(labelList,axis=0)
        uid=np.concatenate(uidList,axis=0)


        if not hasattr(self,'data'):
            self.data=data
            self.label=label
            self.uid=uid

        else:
            if numItemKeep<=0:
                self.data=data
                self.label=label
                self.uid=uid
            else:
                if data.shape[0]<numItemKeep:

                    data=np.r_[self.data,data]
                    label=np.r_[self.label,label]
                    uid=np.r_[self.uid,uid]

                self.data=data[-int(numItemKeep):,:]
                self.label=label[-int(numItemKeep):]
                self.uid=uid[-int(numItemKeep):]

        self.q.close()

    def SetDataLabelUid(self,data,label,uid):
        self.data=data
        self.label=label
        self.uid=uid


    def Label_CSV2Read4Test(self,channels,numCSV):
        self.csv_label={}

        if channels is None:
            channels=self.channels
        else:
            if not isinstance(channels,list):
                channels=[channels]

        channels4Label=[]
        for iChannel in channels:
            channels4Label.extend([x for x in self.channels if iChannel in x])
            channels4Label=list(set(channels4Label))
            channels4Label.sort()


        assert(len(channels4Label)>0),'channels in DataSet.ReadTestSet(channels=...) incorrect... '


        for iChannel in channels4Label:
            iLabel=self.channel2Label[iChannel]
            csvs=[x for x in os.listdir(self.homeCSV+iChannel) if x[-4:]=='.csv']
            if numCSV>0:
                csvs=csvs[0:int(numCSV)]
            for iCSV in csvs:
                self.csv_label[self.homeCSV+iChannel+'/'+iCSV]=iLabel


    def ReadTestSet(self,channels=None,numCSV=0):
        self.Label_CSV2Read4Test(channels=channels,numCSV=numCSV)
        self.q=Queue()

        p=[Process(target=self.ReadCSVOneFile, args=(iLabel,iCSV, self.q,)) for iCSV,iLabel in self.csv_label.items()]
        [ip.start() for ip in p]


    def ReadTestGet(self):

        dataList=[]
        labelList=[]
        uidList=[]

        for i in range(len(self.csv_label)):
            iData, iLabel, iUid,iClass=self.q.get()

            self.dictLabelNumItem[iClass]+=iData.shape[0]

            dataList.append(iData)
            labelList.append(iLabel)
            uidList.append(iUid)


        self.data=np.concatenate(dataList,axis=0)
        self.label=np.concatenate(labelList,axis=0)
        self.uid=np.concatenate(uidList,axis=0)

        self.q.close()



    def ReadTest(self,channels=None,numCSV=0,numProcess=6):
        self.Label_CSV2Read4Test(channels=channels,numCSV=numCSV)



        label_csv=[]
        for iCSV in self.csv_label:
            label_csv.append((self.csv_label[iCSV],iCSV))


        pool = Pool(numProcess,maxtasksperchild=1)

        data_label=pool.starmap(self.ReadCSVOneFile, label_csv)

        dataList=[x[0] for x in data_label]
        labelList=[x[1] for x in data_label]
        uidList=[x[2] for x in data_label]


        self.data=np.concatenate(dataList,axis=0)
        self.label=np.concatenate(labelList,axis=0)
        self.uid=np.concatenate(uidList,axis=0)

        pool.close()



    def __len__(self):
        return len(self.data)


    def __getitem__(self,idx):

        return (self.data[idx,:],self.label[idx,:],self.uid[idx,:])


if __name__=='__main__':
    pass




#
