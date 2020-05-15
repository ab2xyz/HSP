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

from Channel import  Channel


class CSV4DateSet(Channel):
    def __init__(self,homeCSV,channels=None):
        super(CSV4DateSet,self).__init__(homeCSV,channels)

        self._Branch4Train()
        self.__Labels()
        self._Labels()
        self._Data()

        self._DataSplit()




    def _Data(self):
        numCSV={}
        for iChannel in self.channels:
            numCSV[iChannel]=len([x for x in os.listdir(self.homeCSV+iChannel) if x[-4:]=='.csv'])

        self.numCSV=numCSV

        numPID={}
        ratioPID_MC={}

        for iChannel in self.channels:
            with open(self.homeCSV+iChannel+'/runLog') as f:
                f_MC_PID=int(f.readline().split(':')[1].strip())
                f_MC=int(f.readline().split(':')[1].strip())
            numPID[iChannel]=f_MC_PID
            ratioPID_MC[iChannel]=f_MC_PID/f_MC

        self.numPID=numPID
        self.ratioPID_MC=ratioPID_MC

        ratioEvent={}
        maxNumPid=max(self.numPID.values())
        for iChannel in self.channels:
            ratioEvent[iChannel]=maxNumPid/self.numPID[iChannel]
        self.ratioEvent=ratioEvent

    def _DataSplit(self):
        listCSV={}
        numCSV={}

        for iChannel in self.channels:
            iListCSV=[x for x in os.listdir(self.homeCSV+iChannel) if x[-4:]=='.csv']
            iListCSV.sort()
            listCSV[iChannel]=iListCSV
            numCSV[iChannel]=len(iListCSV)

            self.listCSV=listCSV
            self.numCSV=numCSV


    def __Labels(self):

        classes=[]
        for iChannel in self.channels:
            iChannelList=iChannel.split('_')[:3:]
            if iChannelList[0] == iChannelList[1]:
                classes.append('_'.join(iChannelList[0:3:2]))
            elif "DPM" in iChannelList:
                classes.append('_'.join(iChannelList[1:3:]))

        classes=list(set(classes))

        labels=list(range(len(classes)))

        classes.sort()

        self.classes=classes


        self.class_label=dict(zip(self.classes,labels))

        self.numClasses=len(labels)

    def _Labels(self):
        labels={}
        for iChannel in self.channels:
            labels[iChannel]=self.Label(iChannel)
        self.labels=labels


    def Label(self,iChannel):
        iChannelList=iChannel.split('_')[:3:]
        if iChannelList[0] == iChannelList[1]:
            iClass='_'.join(iChannelList[0:3:2])
        else:
            iClass='DPM_'+iChannelList[2]

        return self.class_label[iClass]

    def _Branch4Train(self):
        branchLog=self.homeCSV+'branchLog'
        with open(branchLog,'r') as f:
            while 1:
                f_=f.readline()
                if not f_:
                    break

                if f_[:10]=='branchAll:':
                    f_=f.readline()
                    self.branchAll=f_[1:-3].split(',')


                if f_[:12]=='branch4Train':
                    f_=f.readline()
                    self.branch4Train=f_[1:-3].split(',')

        branchWhitelistHand=self.homeCSV+'branchWhitelistHand'
        branchBlacklistHand=self.homeCSV+'branchBlacklistHand'

        if not os.path.exists(branchWhitelistHand):
            with open(branchWhitelistHand,'w') as f:
                f.close()

        if not os.path.exists(branchBlacklistHand):
            with open(branchBlacklistHand,'w') as f:
                f.close()

        branchWhitelist=self.__GetList_While_Black(iListFile=branchWhitelistHand)
        branchBlacklist=self.__GetList_While_Black(iListFile=branchBlacklistHand)

        for iBranch in branchWhitelist:
            if iBranch in self.branchAll:
                self.branch4Train.append(iBranch)
                branchWhitelist.remove(iBranch)

        for iBranch in branchBlacklist:
            if iBranch in self.branchAll:
                self.branch4Train.remove(iBranch)
                branchBlacklist.remove(iBranch)


        self.branch4TrainNO=list(set(self.branchAll).difference(set(self.branch4Train)))

        self.branchAll.sort()
        self.branch4Train.sort()
        self.branch4TrainNO.sort()

        self.branchTrainLog(creat=True)
        self.branchTrainLog(iList=self.branchAll,iLabel='branchAll')
        self.branchTrainLog(iList=self.branch4Train,iLabel='branch4Train (Train !!!)')
        self.branchTrainLog(iList=self.branch4TrainNO,iLabel='branch4TrainNOUSE')
        self.branchTrainLog(iList=branchWhitelist,iLabel='branchWhitelist NOT exist...')
        self.branchTrainLog(iList=branchBlacklist,iLabel='branchBlacklist NOT exist...')

    def branchTrainLog(self,iList=None,iLabel=None,creat=False):
        if creat:
            with open(self.homeCSV+'branchTrainLog','w') as f:
                f.close()
                return
        with open(self.homeCSV+'branchTrainLog','a') as f:
            f.writelines(iLabel+': '+str(len(iList))+'\n')
            for j in iList:
                f.writelines('{:<20s}'.format(j))

            f.writelines('\n'+'-'*80+'\n')


    def __GetList_While_Black(self,iListFile):
        iList=[]
        with open(iListFile,'r') as f:
            while 1:
                f_=f.readline()
                if not f_:
                    break

                # f_=f_.replace('\n','')

                f__=re.split(',| |\n|;| |/',f_)
                f__=list(set(f__))
                iList.extend(f__)

        iList=list(set(iList))
        if '' in iList:
            iList.remove('')

        iList.sort()

        return iList


    def GetListCSV(self):
        return self.listCSV

    def GetLabels(self):
        return self.labels

    def GetNumClasses(self):
        return self.numClasses

    def GetBranch4Train(self):
        return self.branch4Train


    def Write(self,homeRes):
        import datetime
        import json

        if homeRes.strip()[-1]!='/':
            homeRes=homeRes+'/'

        codeSave=datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')

        jsonName=homeRes+'data'+codeSave+'.json'
        dataDict={'listCSV':self.listCSV}

        with open(jsonName, 'w') as f:
            json.dump(dataDict,f)

        return codeSave



if __name__=='__main__':

    homeCSV='/home/i/iWork/data/csv'

    oCSV4DateSet=CSV4DateSet(homeCSV,channels=None)

    homeRes='/home/i/iWork/data/res'
    codeSave=oCSV4DateSet.Write(homeRes=homeRes)
    print(codeSave)


    print('listCSV')
    print(oCSV4DateSet.GetListCSV())


    print('labels')
    print(oCSV4DateSet.GetLabels())

    print('numClasses')
    print(oCSV4DateSet.GetNumClasses())

    print('branch4Train')
    print(oCSV4DateSet.GetBranch4Train())











#
