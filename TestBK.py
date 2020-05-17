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
from multiprocessing import Pool




class Test():
    def __init__(self,homeCSV,homeRes,NN,codeSave,numFilesCut=20, batchSize=1024, numProcess=1, cuda=True):

        self.homeCSV=homeCSV if homeCSV.strip()[-1]=='/' else homeCSV+'/'
        self.homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'

        self.codeSave=codeSave

        self.numFilesCut=int(numFilesCut)

        self.batchSize=batchSize
        self.numProcess=numProcess
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

        # if self.cuda:
        #     self.NN=self.NN.cuda()
        #
        # self.NN.eval()

        resizeName=self.homeRes+'Resize'+codeSave+'.dat'
        self.resize=np.loadtxt(resizeName).astype(int)




        self.softmax=torch.nn.Softmax(dim=1)


        self.setTest={}
        self.loaderTest={}

        self.uidsTriggerDict={}
        self.uidsUniqueDict={}





    def ReadCSV(self):
        if self.numProcess>0:

            ## 分开处理csvlist： 读取-处理
            pool = Pool(int(numProcess),maxtasksperchild=1)

            channelList= list(self.labels.keys())
            dateSet_loader=pool.map(self.ReadCSV_OneChannel,channelList)
            dateSetList=[x[0] for x in dateSet_loader]
            channelList=[x[1] for x in dateSet_loader]

            pool.close()

            self.setTest=dict(zip(channelList,dateSetList))

            for iChannel in self.setTest:
                self.loaderTest[iChannel]=torch.utils.data.DataLoader(self.setTest[iChannel], batch_size=self.batchSize,  shuffle=False, num_workers=self.numProcess)


        if self.numProcess<=0:

            for iChannel in tqdm(self.labels):
                setTest, iChannel=self.ReadCSV_OneChannel(iChannel)
                self.setTest[iChannel]=setTest
                self.loaderTest[iChannel]=torch.utils.data.DataLoader(self.setTest[iChannel], batch_size=self.batchSize,  shuffle=False, num_workers=self.numProcess)

    def ReadCSV_OneChannel(self,iChannel):
        setTest=DataSet(homeCSV=self.homeCSV,listCSV=self.listCSV4Test,labels=self.labels,numClasses=self.numClasses,branch4Train=self.branch4Train,resize=self.resize,numProcess=0)

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
            iData,iLabel,iUid, iClass=setTest.ReadCSV_OneFile(iClass=iReadClass, iCSV=iReadCSV)

            if data is None:
                data=iData
                label=iLabel
                uid=iUid
            else:
                data=np.r_[data,iData]
                label=np.r_[label,iLabel]
                uid=np.r_[uid,iUid]


        setTest.SetDataLabel(data=data,label=label,uid=uid)


        return (setTest, iChannel)



    def CutEvent(self,channelData,labelTrigger,cutTrigger):

        loader=self.loaderTest[channelData]

        outputNN=None
        uids=None
        for i, data in enumerate(loader,0):
            inputs, iLabel, iUid = data


            if inputs.shape[0]==1:
                continue

            inputs=inputs.float()

            if self.cuda:
                inputs=inputs.cuda()

            outputs = self.NN(inputs)

            outputsSoftmax=self.softmax(outputs)

            outputsSoftmaxTrigger=outputsSoftmax[:,labelTrigger]

            if outputsSoftmaxTrigger.shape[0]==0:
                if uids is None:
                    uids=iUid
                else:
                    uids=torch.cat(uids,iUid)

                continue
            else:

                if outputNN is None:
                    outputNN=outputsSoftmaxTrigger
                    # labels=iLabel
                    uids=iUid
                else:
                    # print(outputNN.shape,outputsSoftmaxTrigger.shape)
                    outputNN=torch.cat((outputNN,outputsSoftmaxTrigger),dim=0)
                    # labels=torch.cat(labels,iLabel)
                    uids=torch.cat((uids,iUid),dim=0)


        if self.cuda:
            uids=uids.cuda()

        outputNNBool=outputNN>cutTrigger
        uidsTrigger=torch.unique(outputNNBool.long()*uids)
        uidsTrigger=uidsTrigger[uidsTrigger.nonzero()]
        uidsUnique=torch.unique(uids)


        uidsTrigger=uidsTrigger.cpu().numpy()
        uidsUnique=uidsUnique.cpu().numpy()
        iEffi=float(len(uidsTrigger))/float(len(uidsUnique))

        return iEffi, uidsTrigger, uidsUnique


    def Test(self,effi=0.99):

        nnDictName=self.homeRes+'NN'+self.codeSave+'.plt'
        self.NN.load_state_dict(torch.load(nnDictName))

        if self.cuda:
            self.NN=self.NN.cuda()

        cutsDict=self.GetCuts(effi=effi)

        [print(x,y) for x,y in cutsDict.items()]

        effiSelDict={}
        effiSelMCTrueDict={}

        for channelData in tqdm(self.labels):

            for jChannel in tqdm(self.labels):
                if jChannel.split('_')[0]==jChannel.split('_')[1]==channelData.split('_')[0]:
                    channelTrigger=jChannel
                    break

            labelTrigger=self.labels[channelTrigger]
            cutTrigger=cutsDict[labelTrigger]

            iEffi, uidsTrigger, uidsUnique=self.CutEvent(channelData=channelData,labelTrigger=labelTrigger,cutTrigger=cutTrigger)

            effiSelMCTrueDict[channelData]=iEffi
            self.uidsTriggerDict[channelData]=uidsTrigger
            self.uidsUniqueDict[channelData]=uidsUnique

            effiMCTrue=self.ReadRunLog2GetEffi(channelData)
            effiSelDict[channelData]=iEffi*effiMCTrue

        effiSelLog=self.homeRes+'effiSel'+self.codeSave+'.log'
        with open(effiSelLog,'w') as f:
            [f.writelines('%s  :  %.6f\n'%(x,y)) for x,y in effiSelDict.items()]


        effiSelLog=self.homeRes+'effiSelMCTrue'+self.codeSave+'.log'
        with open(effiSelLog,'w') as f:
            [f.writelines('%s  :  %.6f\n'%(x,y)) for x,y in effiSelMCTrueDict.items()]


        self.GetEffiBkg()


    def GetEffiBkg(self):
        uidsBkgDictFP={}
        uidsBkgDict={}
        effiBkgDict={}
        numBkg=0

        DPMList=[]

        for iChannel in self.uidsTriggerDict:
            if 'DPM' not in iChannel:
                continue
            DPM_='_'.join(iChannel.split('_')[1:3])
            if DPM_ in DPMList:
                continue
            else:
                DPMList.append(DPM_)

        for iChannel in self.uidsTriggerDict:
            if 'DPM' not in iChannel:
                continue
            for iDPM in DPMList:
                if iDPM in iChannel:
                    iUidsTriggerDict=self.uidsTriggerDict[iChannel]
                    iUidsUniqueDict=self.uidsUniqueDict[iChannel]

                    if iDPM in  uidsBkgDictFP:
                        iUidsTriggerDict=np.r_[uidsBkgDictFP[iDPM],iUidsTriggerDict[:,0]]
                        uidsBkgDictFP[iDPM]=np.unique(iUidsTriggerDict)

                        # if numBkg==0:
                        #     iUidsUniqueDict=self.uidsUniqueDict[iChannel]
                        #     numUidsUniqueDict=iUidsUniqueDict.shape[0]
                        #     iEffiMCTrue=self.ReadRunLog2GetEffi(iChannel)
                        #     numBkg=numUidsUniqueDict/iEffiMCTrue
                        #
                        #     print('*'*80)
                        #     print(iChannel)
                        #     print(iUidsUniqueDict)
                        #     print(numUidsUniqueDict)
                        #     print(iEffiMCTrue)
                        #     print(numBkg)

                        iUidsUniqueDict=np.r_[uidsBkgDict[iDPM],iUidsUniqueDict]
                        uidsBkgDict[iDPM]=np.unique(iUidsUniqueDict)

                    else:
                        uidsBkgDictFP[iDPM]=iUidsTriggerDict[:,0]
                        uidsBkgDict[iDPM]=iUidsUniqueDict

                    # print(uidsBkgDictFP[iDPM].shape)

        for iDPM in uidsBkgDictFP:

            # for iChannel in self.uidsTriggerDict:
            #     if iDPM in iChannel:
            #         numBkg=self.ReadRunLog2GetNumEventMC(iChannel)
            #         break

            effiBkgDict[iDPM]=1.-uidsBkgDictFP[iDPM].shape[0]/uidsBkgDict[iDPM].shape[0]


        effiBkgLog=self.homeRes+'effiBkg'+self.codeSave+'.log'
        with open(effiBkgLog,'w') as f:
            [f.writelines('%s   :  %.6f \n'%(x,y)) for x,y in effiBkgDict.items()]





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


    def ReadRunLog2GetNumEventMC(self,iChannel):
        iLog=self.homeCSV+iChannel+'/runLog'
        with open(iLog,'r') as f:
            while True:
                fLine=f.readline()
                if ('numEventMC' in fLine) and ('numEventMC_PID' not in fLine):
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

                dict_DataSig=list(dict_Uid_DataSig.values())
                dict_DataSig.sort()

                numEvent=float(len(dict_DataSig))

                idxCut=int(numEvent*(1.-effiBkgTarget))
                if idxCut<0:
                    idxCut=0

                iCut=dict_DataSig[idxCut]    # sig>cut  ! ! !


            cutsDict[self.labels[channelSig]]=iCut

        cutLog=self.homeRes+'cut'+self.codeSave+'.log'
        with open(cutLog,'w') as f:
            f.writelines('label  :  cut\n')
            [f.writelines('%d  :  %.8f\n'%(x,y)) for x,y in cutsDict.items()]


        return cutsDict


if __name__=='__main__':
    import time

    homeCSV='/home/i/iWork/data/csv'
    homeRes='/home/i/iWork/data/res'
    from DNN import NN
    codeSave='_20200428_143331'

    numFilesCut=10
    batchSize=1024*16
    numProcess=5
    cuda=True
    effi=0.999


    oTest=Test(homeCSV=homeCSV,homeRes=homeRes,NN=NN,codeSave=codeSave,numFilesCut=numFilesCut, batchSize=batchSize, numProcess=numProcess, cuda=cuda)
    oTest.ReadCSV()
    oTest.Test(effi=effi)
    # oTest.TestMean()

    # for iTime in range(10000):
    #     oTest.Test(effi)
    #     print('\n'+'-'*80+'\n')
    #     time.sleep(200)


    # oTest.GetCuts()


    plt.show()



#
