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


class Test():
    def __init__(self,homeCSV,homeRes,NN, setTest,  codeSave='',batchSize=1024, numProcess=12, cuda=True):

        self.homeCSV=homeCSV

        self.codeSave=codeSave
        self.homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'

        self.NN=NN
        self.setTestBlock=setTest
        self.batchSize=batchSize
        self.numProcess=numProcess
        self.cuda=cuda

        self.channels=self.setTestBlock.channels
        self.channel2Label=self.setTestBlock.channel2Label

        self.setTest={}
        for iChannel in self.channels:
            self.setTest[iChannel]=deepcopy(self.setTestBlock)

        self.softmax=torch.nn.Softmax(dim=1)



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
        self.NNLoad()

        for iChannel in self.channels:

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
        self.trigger_cut={}
        self.trigger_label={}
        for iChannel in self.channels:
            iSplit=iChannel.split('_')
            if iSplit[0]==iSplit[1]:
                channelSig=iChannel
                trigger=iSplit[0]
            else:
                continue

            channelBkg=None
            for jChannel in self.channels:
                if 'DPM' in jChannel:
                    channelBkg=jChannel
                    break
            if channelBkg is None:
                continue

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

            self.trigger_cut[trigger]=iCut
            self.trigger_label[trigger]=labelSig

        cutLog=self.homeRes+'cut'+self.codeSave+'.log'
        with open(cutLog,'w') as f:
            f.writelines('trigger   :   lable  :   cut\n')
            for iTrigger in self.trigger_cut:
                f.writelines('%s   :   %d    :  %.8f\n'%(iTrigger,self.trigger_label[iTrigger],self.trigger_cut[trigger]))


        return self.trigger_cut,self.trigger_label


    def GetTable(self):
        self.channel_effi_Recon={}
        for iChannel in self.channels:
            iSplit=iChannel.split('_')
            trigger=iSplit[0]
            iLabel=self.trigger_label[trigger]
            iCut=self.trigger_cut[trigger]
            if not 'DMP' in iChannel:
                outputs=self.outputsNN[iChannel][:,iLabel]
                uids=self.uids[iChannel]
                uidsUnique=np.unique(uids)
                uidsUniqueLen=len(uidsUnique)

                uidsPositive=(outputs>iCut).astype(int)*uids
                uidsPositiveUnique=np.unique(uidsPositive)
                uidsPositiveUniqueLen=len(uidsPositiveUnique)-1 if 0 in uidsPositiveUnique else len(uidsPositiveUnique)
                self.channel_effi_Recon[iChannel]=uidsPositiveUniqueLen/uidsUniqueLen

            if 'DPM' in iChannel:

                uids=self.uids[iChannel]
                uidsUnique=np.unique(uids)
                uidsUniqueLen=len(uidsUnique)

                uidsPositiveUnique=None
                for iTrigger in self.trigger_label:
                    iLabel=self.trigger_label[iTrigger]
                    iCut=self.trigger_cut[iTrigger]

                    iOutputs=self.outputsNN[iChannel][:,iLabel]
                    iUidsPositive=(iOutputs>iCut).astype(int)*uids
                    iUidsPositive=np.unique(iUidsPositive)

                    if uidsPositiveUnique is None:
                        uidsPositiveUnique=iUidsPositive
                    else:
                        uidsPositiveUnique=np.r_[uidsPositiveUnique,iUidsPositive]
                    uidsPositiveUnique=np.unique(uidsPositive)

                uidsPositiveUniqueLen=len(uidsPositiveUnique)-1 if 0 in uidsPositiveUnique else len(uidsPositiveUnique)
                self.channel_effi_Recon[iChannel]=uidsPositiveUniqueLen/uidsUniqueLen


        effLog=self.homeRes+'effi'+self.codeSave+'.log'
        with open(effLog,'w') as f:
            f.writelines('channel  :  effiRecons\n')
            [f.writelines('%s  :  %.8f\n'%(x,y)) for x,y in self.channel_effi_Recon.items()]


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
