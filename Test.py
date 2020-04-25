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




class Test():
    def __init__(self,homeCSV,homeRes,NN,codeSave,numFilesCut=20,resize=[240]):

        self.homeCSV=homeCSV if homeCSV.strip()[-1]=='/' else homeCSV+'/'
        self.homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'

        self.codeSave=codeSave

        self.numFilesCut=int(numFilesCut)
        # self.batchSize=batchSize

        self.resize=resize


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

        self.NN.eval()

        self.softmax=torch.nn.Softmax(dim=1)


        ##
        numChannel=500
        self.labelsBK=self.labels
        self.labels={}
        counterChannel=0
        for idx in self.labelsBK:
            counterChannel+=1
            if counterChannel>numChannel:
                continue
            self.labels[idx]= self.labelsBK[idx]

        #

        self.ReadCSV()







    def ReadCSV(self):
        self.data={}
        self.label={}
        self.branchSel={}

        for iChannel in tqdm(self.labels):


            data=None
            label=None

            counterCSV=0
            for iCSV in tqdm(self.listCSV4Test[iChannel]):
                counterCSV+=1
                if (self.numFilesCut>0) and (counterCSV>self.numFilesCut):
                    continue

                if not iChannel in self.branchSel:
                    branchAll=pd.read_csv(self.homeCSV+iChannel+'/'+iCSV,nrows=0).columns.tolist()
                    iBranchSel=list(set(branchAll).intersection(set(self.branch4Train)))
                    iBranchSel.sort()
                    self.branchSel[iChannel]=iBranchSel

                iData=pd.read_csv(self.homeCSV+iChannel+'/'+iCSV,usecols=iBranchSel).values


                if data is None:
                    data=iData
                else:
                    data=np.r_[data,iData]

            label=np.ones((data.shape[0]),dtype=np.long)*self.labels[iChannel]

            numData=np.prod(self.resize)
            dataZeros=np.zeros((data.shape[0],numData-data.shape[1]))
            data=np.c_[data,dataZeros]

            dataShape=[data.shape[0]]
            dataShape.extend(self.resize)
            data=np.resize(data,dataShape)

            self.data[iChannel]=data
            self.label[iChannel]=label



    def Test(self):
        for iChannel in tqdm(self.labels):
            inputs=self.data[iChannel]


            inputs=torch.from_numpy(inputs).float()
            outputs=self.softmax(self.NN(inputs))

            outputsMean=outputs.mean(dim=0)

            print('\n')
            print(self.label[iChannel][0])
            print(outputsMean)




if __name__=='__main__':
    from DNN import NN
    codeSave='_20200425_163022'
    homeCSV='/home/i/iWork/data/csv'
    homeRes='/home/i/iWork/data/res'
    resize=[240]

    oTest=Test(homeCSV=homeCSV,homeRes=homeRes,NN=NN,codeSave=codeSave,numFilesCut=1,resize=resize)
    oTest.Test()







#
