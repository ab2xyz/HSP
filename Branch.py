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

from Channel import Channel

class Branch(Channel):
    def __init__(self,homeCSV,channels=None,pidUse=True):
        super(Branch,self).__init__(homeCSV,channels)


        self.pidUse=pidUse


        channelsChecked=[]
        channelsNotChecked=[]
        for iChannel in self.channels:
            iChannelCSV=homeCSV+iChannel+'/'
            iRunLog=iChannelCSV+'runLog'

            if os.path.exists(iRunLog):
                with open(iRunLog, 'r') as f:
                    f_=f.readlines()
                    if "finished\n" in f_:
                        channelsChecked.append(iChannel)
                    else:
                        channelsNotChecked.append(iChannel)
            else:
                channelsNotChecked.append(iChannel)

        channelsChecked.sort()
        channelsNotChecked.sort()

        branchLog=self.homeCSV+'branchLog'
        with open(branchLog,'w') as f:
            f.writelines('Check failed:\n[')
            [f.writelines(x+',') for x in channelsNotChecked]
            f.writelines(']\n\n'+'-'*40+'\n')
            f.writelines('Check successfully:\n[')
            [f.writelines(x+',') for x in channelsChecked]
            f.writelines(']\n\n'+'-'*40+'\n')


        branchAll=[]
        for iChannel in self.channels:
            iChannelCSV=self.homeCSV+iChannel+'/'
            iRunLog=iChannelCSV+'runLog'
            with open(iRunLog, 'r') as f:
                while 1:
                    f_=f.readline()
                    if not f_:
                        break

                    # print(f_)
                    if  f_[0].isdecimal() and f_[-2].isdecimal() and len(f_)>70:
                        f_n=1000
                        while 1:
                            f_=f_.replace('  ',' ')
                            if len(f_)==f_n:
                                break
                            else:
                                f_n=len(f_)
                        f__=f_.split(' ')
                        fBranch=f__[1]
                        branchAll.append(fBranch)

        branchAll=list(set(branchAll))
        branchAll.sort()

        with open(branchLog,'a') as f:
            f.writelines('branchAll: '+str(len(branchAll))+' \n[')
            [f.writelines(x+',') for x in branchAll]
            f.writelines(']\n\n'+'-'*40+'\n')

        branchSel,branchSelNO,branchAllListRemain=self.BranchSel(branchAllList=branchAll)


        with open(branchLog,'a') as f:
            f.writelines('branchSel: '+str(len(branchSel))+' \n[')
            [f.writelines(x+',') for x in branchSel]
            f.writelines(']\n\n'+'-'*40+'\n')

        with open(branchLog,'a') as f:
            f.writelines('branchSelNO: '+str(len(branchSelNO))+' \n[')
            [f.writelines(x+',') for x in branchSelNO]
            f.writelines(']\n\n'+'-'*40+'\n')

        with open(branchLog,'a') as f:
            f.writelines('branchAllListRemain: '+str(len(branchAllListRemain))+' \n[')
            [f.writelines(x+',') for x in branchAllListRemain]
            f.writelines(']\n\n'+'-'*40+'\n')


        branch4Train=branchSel
        branch4Mark=['cand','ev','mode','uid','xmct']

        branch4Train=list(set(branch4Train).difference(set(branch4Mark)))
        branch4Train.sort()
        branch4Mark.sort()



        with open(branchLog,'a') as f:
            f.writelines('branch4Train (Without whilelist unt blacklist): '+str(len(branch4Train))+' \n[')
            [f.writelines(x+',') for x in branch4Train]
            f.writelines(']\n\n'+'-'*40+'\n')

        with open(branchLog,'a') as f:
            f.writelines('branch4Mark : '+str(len(branch4Mark))+' \n[')
            [f.writelines(x+',') for x in branch4Mark]
            f.writelines(']\n\n'+'-'*40+'\n')



        branchBlacklist=self.homeCSV+'branchBlacklistHand'
        branchWhilelist=self.homeCSV+'branchWhilelistHand'

        if not os.path.exists(branchBlacklist):
            with open(branchBlacklist,'w') as f:
                f.close()

        if not os.path.exists(branchWhilelist):
            with open(branchWhilelist,'w') as f:
                f.close()

    def BranchSel(self,branchAllList):
        pidUse=self.pidUse

        branchSel=[]
        branchSelNO=[]
        branchAllListRemain=branchAllList[::]
        for iBranch in branchAllList:

            whiteList=['beamp']
            if iBranch in whiteList:
                branchSel.append(iBranch)
                branchAllListRemain.remove(iBranch)

            useMark = ['mode', 'xmct', 'uid','cand','ev']
            if iBranch in useMark:
                branchSel.append(iBranch)
                branchAllListRemain.remove(iBranch)

            useAvaliable = ['mmiss', 'msum']
            if iBranch in useAvaliable:
                branchSel.append(iBranch)
                branchAllListRemain.remove(iBranch)

            use_mdif='mdif'
            if re.findall("^x(d\d)*"+use_mdif+'$',iBranch):
                branchSel.append(iBranch)
                branchAllListRemain.remove(iBranch)

            use_oang='oang'
            if re.findall("^x(d\d)*"+use_oang+'$',iBranch):
                branchSel.append(iBranch)
                branchAllListRemain.remove(iBranch)


            noteMotherDaughter = ['p', 'tht', 'phi', 'pt', 'pcm', 'thtcm']
            for iNote in noteMotherDaughter:
                if re.findall("^x(d\d)*"+iNote+'$',iBranch):
                    branchSel.append(iBranch)
                    branchAllListRemain.remove(iBranch)

            noteDaughter = ['e', 'mu', 'pi', 'k', 'p', 'max', 'best']
            for iNote in noteDaughter:
                if re.findall("^x(d\d)+pid"+iNote+'$',iBranch):
                    branchSel.append(iBranch)
                    branchAllListRemain.remove(iBranch)

            use_1=['drcnphot','drcthtc','dscnphot','dscthtc','emcecal','emcnb','emcnx','gemnhits','m','muoiron','muonlay','mvddedx','mvdhits','richnphot','richthtc','sttdedx','stthits','tofbeta','tofm2']
            for iNote in use_1:
                if re.findall("^x(d\d)*"+iNote+'$',iBranch):
                    branchSel.append(iBranch)
                    branchAllListRemain.remove(iBranch)


            useVTX = 'pocctau'     # xpocctau  xd0pocctau
            if re.findall("^x(d\d)*"+useVTX+'$',iBranch):
                branchSel.append(iBranch)
                branchAllListRemain.remove(iBranch)

            decang='decang'
            if re.findall("^x(d\d)*"+decang+'$',iBranch):     # xdecang xd0decang
                branchSelNO.append(iBranch)
                branchAllListRemain.remove(iBranch)
            if re.findall("^x(d\d)*c"+decang+'$',iBranch):     # xcdecang xd0cdecang
                branchSel.append(iBranch)
                branchAllListRemain.remove(iBranch)

            use_POC=["pocvx","pocvy","pocvz","pocmag","pocqa","pocdist"]
            for iNote in use_POC:
                if re.findall("^x(d\d)*"+iNote+'$',iBranch):
                    branchSel.append(iBranch)
                    branchAllListRemain.remove(iBranch)


            noUseES=['essumptcl', 'essumptl', 'essumetnl']
            if iBranch.startswith('es'):
                if iBranch not in noUseES:
                    branchSel.append(iBranch)
                else:
                    branchSelNO.append(iBranch)
                branchAllListRemain.remove(iBranch)



            noUse_1 = ['ncand','run']
            if iBranch in noUse_1:
                branchSelNO.append(iBranch)
                branchAllListRemain.remove(iBranch)

            noUse_xmdd=['xm']
            for iNote in noUse_xmdd:
                if re.findall("^"+iNote+'\d\d$',iBranch):
                    branchSelNO.append(iBranch)
                    branchAllListRemain.remove(iBranch)

            noUse_xdal=['xdal']
            for iNote in noUse_xdal:
                if re.findall("^"+iNote+'\d\d$',iBranch):
                    branchSelNO.append(iBranch)
                    branchAllListRemain.remove(iBranch)


            noUse_beam=['beampx','beampy','beampz','beame','beamtht','beamphi','beampt','beamm']
            if iBranch in noUse_beam:
                branchSelNO.append(iBranch)
                branchAllListRemain.remove(iBranch)


            noUse_noteMotherDaughter = ['px','py','pz','e','x','y','z','l','chg','pdg','pxcm','pycm','pzcm','ecm','phicm']
            for iNote in noUse_noteMotherDaughter:
                if re.findall("^x(d\d)*"+iNote+'$',iBranch):

                    branchSelNO.append(iBranch)
                    branchAllListRemain.remove(iBranch)


            noUse_noteDaughter = ['mct','trpdg']
            for iNote in noUse_noteDaughter:
                if re.findall("^x(d\d)+"+iNote+'$',iBranch):
                    branchSelNO.append(iBranch)
                    branchAllListRemain.remove(iBranch)

            noUse_VTX=["altvx","altvy","altvz","vx","vy","vz","len","ctau","ctaud"]
            for iNote in noUse_VTX:
                if re.findall("^x(d\d)*"+iNote+'$',iBranch):
                    branchSelNO.append(iBranch)
                    branchAllListRemain.remove(iBranch)


            noUse_True=["trxpx","trxpy","trxpz","trxe","trxp","trxtht","trxphi","trxpt","trxm"]
            if iBranch in noUse_True:
                branchSelNO.append(iBranch)
                branchAllListRemain.remove(iBranch)

            noUSe_fvxx=['fvxx']
            for iFVXX in noUSe_fvxx:
                if iBranch.startswith(iFVXX):
                    branchSelNO.append(iBranch)
                    branchAllListRemain.remove(iBranch)

            noUse_chi2vx=['chi2vx']
            if iBranch in noUse_chi2vx:
                branchSelNO.append(iBranch)
                branchAllListRemain.remove(iBranch)

        if not pidUse:
            for iPid in branchSel:
                if re.search('pid',iPid):
                    branchSelNO.append(iPid)
                    branchSel.remove(iPid)


        return branchSel,branchSelNO,branchAllListRemain



if __name__=='__main__':
    homeCSV='/home/i/iWork/data/csv'
    channels=None
    pidUse=True

    oBranch=Branch(homeCSV,channels,pidUse)

    branchAllList=pd.read_csv('/home/i/iWork/data/csv/T00_DPM_24/0000000_0002048.csv',nrows=0).columns.tolist()
    print(branchAllList)
    branchSel,branchSelNO,branchAllListRemain=oBranch.BranchSel(branchAllList)
    print(branchSel)
    print(branchSelNO)
    print(branchAllListRemain)
