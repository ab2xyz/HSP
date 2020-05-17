#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import pandas as pd
import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# import torch.nn as nn
from tqdm import  tqdm
# import matplotlib.pyplot as plt
# import json
# from DataSet import DataSet
# from multiprocessing import Pool

from copy import deepcopy
import os


class Test():
    def __init__(self,homeCSV,homeRes,NN, setTest,  codeSave='',batchSize=1024, numProcess=12, cuda=True):

        self.homeCSV=homeCSV

        self.codeSave=codeSave
        self.homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'
        self.homeRes=self.homeRes+codeSave+'/'
        os.makedirs(self.homeRes,exist_ok=True)

        self.NN=NN
        self.batchSize=batchSize
        self.numProcess=numProcess
        self.cuda=cuda
        self.softmax=torch.nn.Softmax(dim=1)


        # self.setTestBlock=setTest
        # self.channels=self.setTestBlock.channels
        # self.channel2Label=self.setTestBlock.channel2Label
        #
        # self.setTest={}
        # for iChannel in self.channels:
        #     self.setTest[iChannel]=deepcopy(self.setTestBlock)


        self.setTest=setTest
        self.channels=list(self.setTest.values())[0].channels
        self.channel2Label=list(self.setTest.values())[0].channel2Label




        ##


        self.outputsNN={}
        self.labels={}
        self.uids={}


    def NNLoad(self):
        nnDictName=self.homeRes+'NN'+self.codeSave+'.plt'
        self.NN.load_state_dict(torch.load(nnDictName))

        if self.cuda:
            self.NN=self.NN.cuda()


    def ReadCSV(self,numCSVReadPerChannel=0):
        print('Read CSV :')
        for iChannel in tqdm(self.channels):
            self.setTest[iChannel].ReadTest(channels=[iChannel],numCSV=numCSVReadPerChannel,numProcess=self.numProcess)

    def RunNN(self,):
        print('RunNN')

        self.NNLoad()

        for iChannel in tqdm(self.channels):

            setTest=self.setTest[iChannel]
            loader=torch.utils.data.DataLoader(setTest, batch_size=self.batchSize,  shuffle=False, num_workers=self.numProcess)

            for i, data in enumerate(loader,0):
                inputs, iLabels, iUid = data

                if inputs.shape[0]==1:
                    continue

                inputs=inputs.float()

                if self.cuda:
                    inputs=inputs.cuda()

                outputs = self.NN(inputs)

                outputSoftmax=self.softmax(outputs)

                if self.cuda:
                    outputSoftmax=outputSoftmax.cpu()
                outputSoftmax=outputSoftmax.detach()

                if not iChannel in self.outputsNN:
                    self.outputsNN[iChannel]=outputSoftmax
                    self.labels[iChannel]=iLabels
                    self.uids[iChannel]=iUid
                else:
                    self.outputsNN[iChannel]=torch.cat((self.outputsNN[iChannel],outputSoftmax),dim=0)
                    self.labels[iChannel]=torch.cat((self.labels[iChannel],iLabels),dim=0)
                    self.uids[iChannel]=torch.cat((self.uids[iChannel],iUid),dim=0)

            self.outputsNN[iChannel]=self.outputsNN[iChannel].numpy()
            self.labels[iChannel]=self.labels[iChannel].numpy()
            self.uids[iChannel]=self.uids[iChannel].numpy()


    def GetCuts(self,effiBkgTarget):
        print('GetCuts')

        self.trigger_cut={}
        self.trigger_label={}
        for iChannel in tqdm(self.channels):
            iSplit=iChannel.split('_')
            if iSplit[0]==iSplit[1]:
                channelSig=iChannel
                trigger=iSplit[0]
            else:
                continue

            channelBkg=None
            for jChannel in self.channels:
                if ('DPM' in jChannel) and (jChannel[:jChannel.find('_')]==trigger):
                    channelBkg=jChannel
                    break
            if channelBkg is None:
                continue

            # print(channelSig,channelBkg)

            effiBkgReconstruction=self.ReadRunLog2GetEffi(channelBkg)
            effiBkgReconstruction4Cut=(1-effiBkgTarget)/effiBkgReconstruction

            # effiBkgReconstruction4Cut=1-effiBkgTarget    ###


            labelSig=self.channel2Label[channelSig]    # 1
            labelBkg=self.channel2Label[channelBkg]  # 0


            # outputSig=self.outputsNN[channelSig][:,labelSig]
            outputBkg=self.outputsNN[channelBkg][:,labelSig]

            # uidsSig=self.uids[channelSig]
            uidsBkg=self.uids[channelBkg]

            # labelsSig=self.labels[channelSig]
            # labelsBkg=self.labels[channelBkg]

            outputBkgArgsort=np.argsort(outputBkg)
            outputBkgSort=outputBkg[outputBkgArgsort]
            uidsBkgSort=uidsBkg[outputBkgArgsort]

            dictBkgSort=dict(zip(uidsBkgSort,outputBkgSort))

            bkgSort=list(dictBkgSort.values())
            bkgSort.sort()



            numEvent=float(len(bkgSort))
            idxCut=int(numEvent*(1.-effiBkgReconstruction4Cut))
            if idxCut<0:
                idxCut=0




            iCut=bkgSort[idxCut]    # sig>cut  ! ! !

            # print(idxCut,iCut)

            self.trigger_cut[trigger]=iCut
            self.trigger_label[trigger]=labelSig

        cutLog=self.homeRes+'cut'+self.codeSave+'.log'
        with open(cutLog,'w') as f:
            f.writelines('trigger   :   lable  :   cut\n')
            for iTrigger in self.trigger_cut:
                f.writelines('%s   :   %d    :  %.8f\n'%(iTrigger,self.trigger_label[iTrigger],self.trigger_cut[iTrigger]))


        return self.trigger_cut,self.trigger_label


    def GetTable(self):
        print('GetTable')
        self.effiReconstruction={}
        self.effiTriggerReconstruction={}
        self.effiTriggerRaw={}
        for iChannel in tqdm(self.channels):
            iSplit=iChannel.split('_')
            trigger=iSplit[0]
            iLabel=self.trigger_label[trigger]
            iCut=self.trigger_cut[trigger]


            outputs=self.outputsNN[iChannel][:,iLabel]
            uids=self.uids[iChannel]
            uidsUnique=np.unique(uids)
            uidsUniqueLen=len(uidsUnique)

            uidsPositive=(outputs>iCut).astype(int)*uids
            uidsPositiveUnique=np.unique(uidsPositive)
            uidsPositiveUniqueLen=len(uidsPositiveUnique)-1 if 0 in uidsPositiveUnique else len(uidsPositiveUnique)
            self.effiTriggerReconstruction[iChannel]=uidsPositiveUniqueLen/uidsUniqueLen

            self.effiReconstruction[iChannel]=self.ReadRunLog2GetEffi(iChannel)
            self.effiTriggerRaw[iChannel]=self.effiTriggerReconstruction[iChannel]*self.effiReconstruction[iChannel]

        effiLog=self.homeRes+'effi'+self.codeSave+'.log'
        with open(effiLog,'w') as f:
            f.writelines('%-30s %-30s %-30s %-30s\n'%('channel','effiReconstruction', 'effiTriggerReconstruction', 'effiTriggerRaw'))
            for iChannel in self.channels:
                f.writelines('%-30s      %-20.10f    %-20.10f    %-20.10f\n'%(iChannel,self.effiReconstruction[iChannel],self.effiTriggerReconstruction[iChannel],self.effiTriggerRaw[iChannel]))


    def GetDPMPerformance(self):
        print('GetDPMPerformance')
        self.effiDPM={}

        self.channelsDPM=[]
        for iChannel in self.channels:
            if 'DPM' in iChannel:
                iChannelDPM=iChannel[iChannel.find('_')+1:]
                if not iChannelDPM  in self.channelsDPM:
                    self.channelsDPM.append(iChannelDPM)
        self.channelsDPM.sort()

        for iChannelDPM in tqdm(self.channelsDPM):

            uidsUniqueLen=None
            uidsPositiveUnique=None
            for iChannel in self.channels:
                if not iChannelDPM in iChannel:
                    continue

                iSplit=iChannel.split('_')
                trigger=iSplit[0]
                iLabel=self.trigger_label[trigger]
                iCut=self.trigger_cut[trigger]

                uids=self.uids[iChannel]
                if uidsUniqueLen is None:
                    uidsUniqueLen=self.ReadRunLog2GetNumEventMC(iChannel)

                iOutputs=self.outputsNN[iChannel][:,iLabel]
                iUidsPositive=(iOutputs>iCut).astype(int)*uids
                iUidsPositiveUnique=np.unique(iUidsPositive)

                if uidsPositiveUnique is None:
                    uidsPositiveUnique=iUidsPositive
                else:
                    uidsPositiveUnique=np.r_[uidsPositiveUnique,iUidsPositive]
                    uidsPositiveUnique=np.unique(uidsPositiveUnique)

            uidsPositiveUnique=uidsPositiveUnique[uidsPositiveUnique!=0]

            uidsPositiveUniqueLen=len(uidsPositiveUnique)
            self.effiDPM[iChannelDPM]=uidsPositiveUniqueLen/uidsUniqueLen

        dpmLog=self.homeRes+'effiDPM'+self.codeSave+'.log'
        with open(dpmLog,'w') as f:
            for iChannelDPM in self.effiDPM:
                f.writelines('%s\n'%iChannelDPM)
                f.writelines('%s :  %.6f \n'%('Effi',self.effiDPM[iChannelDPM]))
                f.writelines('%s  :  %d \n'%('numEventMC',uidsUniqueLen))
                f.writelines('%s  :  %d \n'%('FalsePositive (Uids below)',uidsPositiveUniqueLen))
                [f.writelines('%s '%x) for x in uidsPositiveUnique]
                f.writelines('\n'+'-'*80+'\n')





    def ReadRunLog2GetNumEventMC(self,iChannel):
        iLog=self.homeCSV+iChannel+'/runLog'
        with open(iLog,'r') as f:
            while True:
                fLine=f.readline()
                if ('numEventMC' in fLine) and ('numEventMC_PID' not in fLine):
                    iEff=float(fLine.split(':')[1])
                    break
            return iEff



    def ReadRunLog2GetEffi(self,iChannel):
        iLog=self.homeCSV+iChannel+'/runLog'
        with open(iLog,'r') as f:
            while True:
                fLine=f.readline()
                if 'numEventMC_PID/numEventMC' in fLine:
                    iEff=float(fLine.split(':')[1])
                    break
            return iEff









#
