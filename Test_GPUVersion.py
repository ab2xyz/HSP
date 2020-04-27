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


        ##




        ##

        # numLabels=1500
        # self.labelsBK=self.labels
        # self.labels={}
        # counterLabers=0
        # for i in self.labelsBK:
        #     print(i)
        #     counterLabers+=1
        #     if counterLabers>numLabels:
        #         break
        #
        #     self.labels[i]=self.labelsBK[i]
        # self.numClasses=len(set(list(self.labels.values())))
        # print(self.numClasses)
        #
        # # print(self.labelsBK)
        # print(self.labels)


        ##









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
                    labels=labels.cuda()
                    uids=uids.cuda()

                outputs = self.NN(inputs)

                softmaxOutputs=self.softmax(outputs)


                print(softmaxOutputs.shape,labels.shape,uids.shape)
                iOutputNN=torch.concat([softmaxOutputs,labels,uids],dim=0)

                if outputNN is None:
                    outputNN=iOutputNN
                else:
                    outputNN=torch.concat([outputNN,iOutputNN],dim=1)




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





    def GetCuts(self,effi=0.9):

        outputNN=None

        idBkg=[]
        idSig=[]
        idTrigger=[]
        for iChannel in tqdm(self.labels):
            iChannleSplit=iChannel.split('_')

            if 'DPM'  in iChannel:
                idBkg.append(self.labels[iChannel])
            elif iChannleSplit[0]==iChannleSplit[1]:
                idSig.append(self.labels[iChannel])

                idTrigger.append(int(iChannel.split('_')[0][1:]))

            else:
                continue




            loader=self.loaderTest[iChannel]

            lossTest=0.
            for i, data in enumerate(loader,0):
                inputs, labels, uids = data


                if inputs.shape[0]==1:
                    continue

                inputs=inputs.float()
                labels=torch.unsqueeze(labels,1)
                labels=labels.float()
                uids=uids.float()

                if self.cuda:
                    inputs=inputs.cuda()
                    labels=labels.cuda()
                    uids=uids.cuda()

                outputs = self.NN(inputs)

                softmaxOutputs=self.softmax(outputs)


                print(softmaxOutputs.shape,labels.shape,uids.shape)

                iOutputNN=torch.cat([softmaxOutputs,labels,uids],dim=1)

                if outputNN is None:
                    outputNN=iOutputNN
                else:
                    outputNN=torch.cat([outputNN,iOutputNN],dim=0)



        print(outputNN)
        print(outputNN.shape)
        print(type(outputNN))


        lb=np.zeros(len(idSig))+1e-8
        ub=np.ones(len(idSig))-1e-8


        cuts=[0.5,0.5,0.5,0.5]

        self.FuncCuts(cuts, outputNN=outputNN,   idBkg=idBkg, idSig=idSig,effi=effi)




        ##

        popuMemo, popuMemoMarks,popuRecommand=self.PSOND(self.FuncCuts,lb,ub, swarmsize=200,maxiter=100, memorysize=1000,pltShow=True, outputNN=outputNN,idBkg=idBkg,idSig=idSig,effi=effi)

        popuBest=popuMemo[np.argmin(np.abs(popuMemoMarks[:,0])),:]

        cuts=popuBest



        cutsDict=dict(zip(idTrigger,cuts))





        cutsLog=self.homeRes+'cuts'+self.codeSave+'.log'
        with open(cutsLog,'w') as f:
            [f.writelines('%s  :  %.6f\n'%(x,y)) for x,y in cutsDict.items()]

        return cutsDict





    def FuncCuts(self,cuts, **kwargs):
        outputNN=kwargs['outputNN']
        idBkg=kwargs['idBkg']
        idSig=kwargs['idSig']
        effi=kwargs['effi']

        ## --  Bkg

        bkgIdx=None     # All the idx of bkg
        for iBkg in idBkg:
            iBkgIdx=outputNN[:,-2]==iBkg
            if bkgIdx is None:
                bkgIdx=iBkgIdx
            else:
                bkgIdx=bkgIdx +iBkgIdx

        bkgItem=outputNN[bkgIdx,-1]     # All the  items of bkg
        bkgUid=torch.unique(bkgItem)     # The uid of bkg
        bkgNum=bkgUid.shape[0]       # The number of bkg




        ## -- Sig
        cuts=cuts[:len(idSig)]

        sigIdx=None     # All the index > cuts  True signal
        idxCut=-1
        for iSig in idSig:
            idxCut+=1
            iCut=cuts[idxCut]

            iSigIdx=outputNN[:,iSig]>iCut

            if sigIdx is None:
                sigIdx=iSigIdx
            else:
                sigIdx=sigIdx+iSigIdx


        #



        sigIdxUid=sigIdx.float()*outputNN[:,-1]

        bkgUidTrue=np.setdiff1d(bkgUid,sigIdxUid)

        bkgTrueNum=bkgUidTrue.shape[0]

        bkgEff=float(bkgTrueNum)/float(bkgNum)-1e-15

        effiErrBkg=np.log((1.-effi)/(1.-bkgEff))**2


        cutsNorm=np.linalg.norm(cuts,ord=2)

        return effiErrBkg,cutsNorm


    def PSOND(self,Func,lb,ub, swarmsize=500,maxiter=100, memorysize=1000,pltShow=True, *args,  **kwargs):
        lb,ub=np.array(lb),np.array(ub)

        mb=(lb+ub)/2

        numB=len(mb)

        target=Func(mb, *args, **kwargs)

        numTarget = len(target)


        lwcc=np.ones((numTarget+3))*0.001
        uwcc=np.ones((numTarget+3))*3.

        lb=np.r_[lb,lwcc]
        ub=np.r_[ub,uwcc]

        mb=(lb+ub)/2
        numB=len(mb)


        for iIter in tqdm(range(maxiter)):
            # print(iIter)
            if iIter==0:

                popu=np.random.random((swarmsize//1,numB))*(ub-lb)+lb
                vPopu=np.random.random((swarmsize//1,numB))*(ub-lb)*0.01


                popuMemo=np.zeros((memorysize//1+swarmsize//1,numB))
                popuMemoMarks=np.ones((memorysize//1+swarmsize//1,numTarget))*1e15


                popuLocal=np.zeros((swarmsize//1,numB,numTarget))
                popuLocalMarks=np.ones((swarmsize//1,numTarget))*1e15

                numMemo=0

            else:

                idx0=np.random.choice(range(numMemo),size=swarmsize)
                idx1=np.random.choice(range(numMemo),size=swarmsize)
                idx2=np.random.choice(range(numMemo),size=swarmsize)
                idx3=np.random.choice(range(numMemo),size=swarmsize)

                r=[]
                for iR in range(numTarget+2):
                    r.append(np.random.random()/(numTarget+2))


                wcc=[]
                nRepeat=vPopu.shape[1]
                for iWcc in range(-(numTarget+3),0):
                    wcc.append(popu[:,iWcc][:,np.newaxis].repeat(nRepeat,axis=1))

                idx1=np.random.choice(range(numMemo),size=swarmsize)
                idx2=np.random.choice(range(numMemo),size=swarmsize)

                vPopu=wcc[-(numTarget+3)]*vPopu+r[0]*wcc[-(numTarget+2)]*(popuMemo[idx1,:]-popu)+r[1]*wcc[-(numTarget+1)]*(popuMemo[idx2,:]-popu)

                for iNumTarget in range(numTarget):
                    vPopu+=r[iNumTarget+2]*wcc[-(numTarget-iNumTarget)]*(popuLocal[:,:,iNumTarget]-popu)


                popu=popu+vPopu


                for idxB in range(numB):
                    popu[popu[:,idxB]<lb[idxB],idxB]=lb[idxB]
                    popu[popu[:,idxB]>ub[idxB],idxB]=ub[idxB]


            for iSwarm in range(swarmsize//1):

                target=Func(popu[iSwarm,:-3], *args, **kwargs)

                tar=np.array(target)
                tarFlag=((tar>popuMemoMarks[:numMemo]).all(axis=1)).any()

                if tarFlag:
                    # Be dominated
                    continue

                tarFlag=~(tar<popuMemoMarks[:numMemo]).all(axis=1)

                popuMemoMarksBK=popuMemoMarks[:numMemo][tarFlag,:].copy()
                popuMemoBK=popuMemo[:numMemo][tarFlag,:].copy()

                numMemo=popuMemoMarksBK.shape[0]+1
                popuMemoMarks[:numMemo-1,:]=popuMemoMarksBK.copy()
                popuMemo[:numMemo-1,:]=popuMemoBK.copy()


                popuMemoMarks[numMemo-1,:]=tar
                popuMemo[numMemo-1,:]=popu[iSwarm,:].copy()



                ##

                for iTarget in range(numTarget):
                    if tar[iTarget]<=popuLocalMarks[iSwarm,iTarget]:
                        popuLocalMarks[iSwarm,iTarget]=tar[iTarget]
                        popuLocal[iSwarm,:,iTarget]=popu[iSwarm,:].copy()


            if numMemo>memorysize:     #delete dense particles
                num=numMemo-memorysize
                for iDel in range(num):
                    idxRand=np.random.choice(range(numMemo),3)

                    iR=[]
                    for iRand in idxRand:
                        popuMemoMarksNorm=(popuMemoMarks-popuMemoMarks[:numMemo,:].min(axis=0))/popuMemoMarks[:numMemo,:].max(axis=0)

                        iMark=popuMemoMarksNorm[iRand,:]
                        iMarkDiff=popuMemoMarksNorm-iMark


                        iNorm=(iMarkDiff**2).sum(axis=1)
                        iNorm.sort()
                        iR.append(iNorm[:10].sum())

                    idx=idxRand[np.argmin(iR)]

                    popuMemoMarks[idx,:]=popuMemoMarks[numMemo-1,:]
                    popuMemo[idx,:]=popuMemo[numMemo-1,:]

                    numMemo-=1

                print(numMemo, memorysize)





            if pltShow:

                plt.figure('Fit')
                plt.clf()
                plt.plot(popuMemoMarks[:numMemo,0],popuMemoMarks[:numMemo,1],'.')

                plt.grid()
                plt.title(numMemo)
                plt.pause(0.01)




        if 'popuRecommand' not in locals():


            popuRecommand=popuMemo[np.argmin(np.linalg.norm(popuMemoMarks[:numMemo,:],axis=1,ord=2)),:-(numTarget+3)]

        popuMemo=popuMemo[:numMemo,:-(numTarget+3)]
        popuMemoMarks=popuMemoMarks[:numMemo,:]
        popuRecommand=popuRecommand


        print(popuMemo.shape,popuMemoMarks.shape)
        return popuMemo, popuMemoMarks,popuRecommand




if __name__=='__main__':
    import time

    homeCSV='/home/i/iWork/data/csv'
    homeRes='/home/i/iWork/data/res'
    from DNN import NN
    codeSave='_20200426_200040'

    numFilesCut=2
    batchSize=1024*16
    numProcessor=1
    cuda=True
    effi=0.999


    oTest=Test(homeCSV=homeCSV,homeRes=homeRes,NN=NN,codeSave=codeSave,numFilesCut=numFilesCut, batchSize=batchSize, numProcessor=numProcessor, cuda=cuda)
    # oTest.Test()
    # oTest.TestMean()

    # for iTime in range(10000):
    #     oTest.Test(effi)
    #     print('\n'+'-'*80+'\n')
    #     time.sleep(200)


    oTest.GetCuts()


    plt.show()



#
