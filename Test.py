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
from DataSet import DataSet




class Test():
    def __init__(self,homeCSV,homeRes,NN,codeSave,numFilesCut=20, batchSize=1024, numProcessor=1, cuda=True):

        self.homeCSV=homeCSV if homeCSV.strip()[-1]=='/' else homeCSV+'/'
        self.homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'

        self.codeSave=codeSave

        self.numFilesCut=int(numFilesCut)

        self.batchSize=batchSize
        self.numProcessor=numProcessor
        self.cuda=cuda


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

        if self.cuda:
            self.NN=self.NN.cuda()

        resizeName=self.homeRes+'Resize'+codeSave+'.dat'
        self.resize=np.loadtxt(resizeName).astype(int)


        self.NN.eval()

        self.softmax=torch.nn.Softmax(dim=1)


        self.setTest={}
        self.loaderTest={}







        self.ReadCSV()






    def ReadCSV(self):

        for iChannel in tqdm(self.labels):
            self.setTest[iChannel]=DataSet(homeCSV=self.homeCSV,listCSV=self.listCSV4Test,labels=self.labels,numClasses=self.numClasses,branch4Train=self.branch4Train,resize=self.resize,numProcess=self.numProcessor)

            data=None
            label=None
            uid=None

            counterCSV=0
            for iCSV in tqdm(self.listCSV4Test[iChannel]):
                counterCSV+=1
                if (self.numFilesCut>0) and (counterCSV>self.numFilesCut):
                    continue

                iReadCSV=self.homeCSV+iChannel+'/'+iCSV
                iReadClass=self.labels[iChannel]
                iData,iLabel,iUid, iClass=self.setTest[iChannel].ReadCSV_OneFile(iClass=iReadClass, iCSV=iReadCSV)

                if data is None:
                    data=iData
                    label=iLabel
                    uid=iUid
                else:
                    data=np.r_[data,iData]
                    label=np.r_[label,iLabel]
                    uid=np.r_[uid,iUid]


            self.setTest[iChannel].SetDataLabel(data=data,label=label,uid=uid)
            self.loaderTest[iChannel]=torch.utils.data.DataLoader(self.setTest[iChannel], batch_size=self.batchSize,  shuffle=False, num_workers=self.numProcessor)



    def Test(self,effi=0.9):

        nnDictName=self.homeRes+'NN'+self.codeSave+'.plt'
        self.NN.load_state_dict(torch.load(nnDictName))

        if self.cuda:
            self.NN=self.NN.cuda()

        cutsDict=self.GetCuts(effi=effi)

        effiSelDict={}

        for iChannel in tqdm(self.labels):

            loader=self.loaderTest[iChannel]

            outputNN=None
            for i, data in enumerate(loader,0):
                inputs, labels, uids = data


                if inputs.shape[0]==1:
                    continue

                inputs=inputs.float()

                if self.cuda:
                    inputs=inputs.cuda()

                outputs = self.NN(inputs)

                softmaxOutputs=self.softmax(outputs)

                if self.cuda:
                    softmaxOutputs=softmaxOutputs.cpu()
                softmaxOutputsNumpy=softmaxOutputs.detach().numpy()

                iOutputNN=np.c_[softmaxOutputsNumpy,labels,uids]

                if outputNN is None:
                    outputNN=iOutputNN
                else:
                    outputNN=np.r_[outputNN,iOutputNN]




            iTrigger=int(iChannel.split('_')[0][1:])
            iCut=cutsDict[iTrigger]


            uid=outputNN[:,-1]
            uidUnique=np.unique(uid)
            uidUniqueNum=uidUnique.shape[0]


            for jChannel in tqdm(self.labels):
                if jChannel.split('_')[0]==jChannel.split('_')[1]==iChannel.split('_')[0]:
                    iClass=self.labels[jChannel]
                    break

            # iClass=self.labels[]

            iSelect=outputNN[:,iClass]>iCut
            iSelectUid=uid*iSelect.astype(int)
            iSelectUidIntersect=np.unique(np.intersect1d(uidUnique,iSelectUid))
            uidUniqueNumSel=iSelectUidIntersect.shape[0]

            effiSelDict[iChannel]=float(uidUniqueNumSel)/float(uidUniqueNum)


        effiSelLog=self.homeRes+'effiSel'+self.codeSave+'.log'
        with open(effiSelLog,'w') as f:
            [f.writelines('%s  :  %.6f\n'%(x,y)) for x,y in effiSelDict.items()]







    def TestMean(self):

        for iChannel in tqdm(self.labels):
            loader=self.loaderTest[iChannel]

            lossTest=0.
            for i, data in enumerate(loader,0):
                inputs, labels, uids = data


                if inputs.shape[0]==1:
                    continue

                inputs=inputs.float()
                labels=labels.long()

                if self.cuda:
                    inputs=inputs.cuda()
                    labels=labels.cuda()

                outputs = self.NN(inputs)


                if i==0:
                    outputsMean=self.softmax(outputs).mean(dim=0)
                    print('\n',labels[0],outputsMean,iChannel)



    def ReadRunLog2GetEffi(self,iChannel):
        iLog=self.homeCSV+iChannel+'/runLog'
        with open(iLog,'r') as f:
            while True:
                fLine=f.readline()
                if 'numEventMC_PID/numEventMC' in fLine:
                    iEff=float(fLine.split(':')[1])
                    break
            return iEff

    def GetCuts(self,effi=0.999):
        cutsDict={}
        for channelSig in tqdm(self.labels):
            channelSigSplit=channelSig.split('_')
            if channelSigSplit[0]!=channelSigSplit[1]:
                continue

            for channelBkg in tqdm(self.labels):
                channelBkgSplit=channelBkg.split('_')
                if channelBkgSplit[0]!=channelSigSplit[0]:
                    continue
                if not 'DPM' in channelBkg:
                    continue

                loaderSig=self.loaderTest[channelSig]
                loaderBkg=self.loaderTest[channelBkg]


                effiSig=self.ReadRunLog2GetEffi(channelSig)
                effiBkg=self.ReadRunLog2GetEffi(channelBkg)

                effiBkgTarget=(1-effi)/effiBkg


                outputNN=None
                for loader in [loaderSig,loaderBkg]:

                    for i, data in enumerate(loader,0):
                        inputs, iLabels, iUid = data

                        if inputs.shape[0]==1:
                            continue

                        inputs=inputs.float()

                        if self.cuda:
                            inputs=inputs.cuda()

                        outputs = self.NN(inputs)

                        softmaxOutputs=self.softmax(outputs)


                        if outputNN is None:
                            outputNN=softmaxOutputs
                            labels=iLabels
                            uids=iUid
                        else:
                            outputNN=torch.cat([outputNN,softmaxOutputs],dim=0)
                            labels=torch.cat([labels,iLabels],dim=0)
                            uids=torch.cat([uids,iUid],dim=0)


                if self.cuda:
                    outputNN=outputNN.cpu()
                outputNN=outputNN.detach().numpy()
                labels=labels.detach().numpy()
                uids=uids.detach().numpy()


                classSig=self.labels[channelSig]
                classBkg=self.labels[channelBkg]

                dataSig=outputNN[:,classSig]
                # dataBkg=outputNN[:,classBkg]

                # boolSig=labels==classSig
                boolBkg=labels==classBkg

                dataSig_Bkg=dataSig[boolBkg]   # BKG's sig marks
                uidBkg=uids[boolBkg]

                dataSig_Bkg_Argsort=np.argsort(dataSig_Bkg)

                dataSig_Bkg_Sort=dataSig_Bkg[dataSig_Bkg_Argsort]    # dataSig_Bkg Sort
                uidBkg_Sort=uidBkg[dataSig_Bkg_Argsort]     #  keys sorted with Origin-value-sorting...


                dict_Uid_DataSig=dict(zip(uidBkg_Sort,dataSig_Bkg_Sort))

                dict_DataSig=list(dict_Uid_DataSig.values()).sort()
                # dict_DataSig.sort()
                # numEvent=float(len(dict_DataSig))

                # idxCut=int(numEvent*(1.-effiBkgTarget))
                # if idxCut<0:
                #     idxCut=0

                iCut=dict_DataSig[max(0,int(float(len(dict_DataSig))*(1.-effiBkgTarget)))]    # sig>cut  ! ! !



            cutsDict[self.labels[channelSig]]=iCut

        [print(x,y) for x,y in cutsDict.items()]

        return cutsDict


if __name__=='__main__':
    import time

    homeCSV='/home/i/iWork/data/csv'
    homeRes='/home/i/iWork/data/res'
    from DNN import NN
    codeSave='_20200426_200040'

    numFilesCut=1
    batchSize=1024*16
    numProcessor=1
    cuda=True
    effi=0.999


    oTest=Test(homeCSV=homeCSV,homeRes=homeRes,NN=NN,codeSave=codeSave,numFilesCut=numFilesCut, batchSize=batchSize, numProcessor=numProcessor, cuda=cuda)
    oTest.Test()
    # oTest.TestMean()

    # for iTime in range(10000):
    #     oTest.Test(effi)
    #     print('\n'+'-'*80+'\n')
    #     time.sleep(200)


    # oTest.GetCuts()


    plt.show()



#
