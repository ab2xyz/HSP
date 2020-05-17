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


class CSV4DataSet(Channel):
    def __init__(self,homeCSV,channels=None):
        '''
        1. GetBranch4Train
            : self.branch4Train
            : write branches to log: branchLog
        2. GetLabels
        3. GetNumClasses
        '''

        super(CSV4DataSet,self).__init__(homeCSV,channels)

        self.Branch4Train()


        print(self.channels)

    def Branch4Train(self):
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


    def Labels(self):
        '''
        Calculate labels
        '''






if __name__=='__main__':
    homeCSV='/home/i/IGSI/data/data/csv/train'
    channels=None
    oCSV4DataSet=CSV4DataSet(homeCSV=homeCSV,channels=channels)









#
