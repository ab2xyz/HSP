#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uproot
import pandas as pd
import os
from multiprocessing import Process
from tqdm import tqdm
import time
import shutil


from Channel import Channel

class Root2CSV(Channel):
    def __init__(self,homeRoot,homeCSV,channels=None,numEventPerFile=2048,numProcess=10):
        super(Root2CSV,self).__init__(homeCSV,channels)


        # home of Root
        self.homeRoot=homeRoot
        if self.homeRoot.strip()[-1]!='/':
            self.homeRoot=self.homeRoot+'/'

        self.channels=self.Channel(home=self.homeRoot,channels=channels)

        self.flagChecked=False

        # # home of csv:
        os.makedirs(self.homeCSV,exist_ok=True)

        self.numProcess=numProcess


        # self.channels.sort()
        self.channels=self.channels[::-1]

        # number of channels
        self.numChannels=len(self.channels)

        # numEventPerFile
        self.numEventPerFile=numEventPerFile

    def ChangeChannelName(self):
        '''
        Rename the channels from "T00_DPM_24" to "T00_24_DPM" with format "Method_Energy_Data"
        '''

        channels=os.listdir(self.homeRoot)
        channels.sort()
        for iChannel in channels:
            jChannelList=iChannel.split('_')
            jChannelList[1],jChannelList[2]=jChannelList[2],jChannelList[1]
            iStr='_'
            jChannel=iStr.join(jChannelList)
            os.rename(self.homeRoot+iChannel,self.homeRoot+jChannel)   #     return

    def LogGetNumEventMC(self,root):  #     return numEventMC
        iLog=root.replace('.root','.log')

        numEventMC=0
        try:
            with open(iLog,'r',errors='ignore') as f:
                l=1
                while l:
                    l=f.readline()
                    if l.startswith('[PndSimpleCombinerTask] evt '):
                        numEventMC=int(l[28:])
        except:
            pass

        return numEventMC

    def RootGetTree(self,root):   #        return tree
        treeList = list(set([x for x in [x.decode("utf-8").split(';')[0] for x in uproot.open(root).keys()] if x.startswith('ntp')]))
        treeList.sort()
        tree=treeList[-1]
        return tree

    def RootGetBranch(self,root,tree):
        branches = [x.decode("utf-8") for x in uproot.open(root)[tree].keys()]
        branches.sort()
        return branches     # return branches

    def RootGetData(self,root,tree,branches):   # return data
        data= uproot.open(root)[tree].pandas.df(branches).dropna(axis=0, how='any')
        return data

    def RootRead_Process_Write_Channel(self,channel):
        '''
        Read Process Write for one channel
        '''


        # runLog : if the channel is in process.   ==0  : in process
        #                                                                               !=0  : in process :  finished
        iChannel=self.homeRoot+channel+'/'


        iChannelCSV=self.homeCSV+channel+'/'

        runLog=iChannelCSV+'runLog'

        # Crate the csv - channel folder:
        os.makedirs(iChannelCSV, exist_ok=True)

        # check the channel in process or not. if it finished, then return
        if os.path.exists(runLog):
            return

        # If the channel is not processed, then do it.
        with open(runLog,'w') as f:
            f.close()



        roots=[x for x in os.listdir(iChannel) if len(x)>5 and x[-5:]=='.root']
        roots.sort()


        # Determine whether it is a signal or a noise


        iRoot=roots[0].split('_')
        nRoot=len(iRoot)
        IRoot=list(set(iRoot))
        NRoot=len(IRoot)
        if nRoot!=NRoot:
            sig=True
        else:
            sig=False

        # Process root files
        for iRoot in roots:
            try:
                iTree=self.RootGetTree(iChannel+iRoot)
                iBranch=self.RootGetBranch(iChannel+iRoot,iTree)
                iData=self.RootGetData(iChannel+iRoot,iTree,iBranch)
            except :
                continue

            iNumEventMC=self.LogGetNumEventMC(iChannel+iRoot)

            # check if it is sig
            if sig:
                iData=iData[iData['xmct']>0.5]
            else:
                iData=iData[iData['xmct']<0.5]

            # concat data
            if 'oData' not in locals():
                oData=iData
            else:
                oData=pd.concat((oData,iData))

            if "oNumEventMC" not in locals():
                oNumEventMC=iNumEventMC
            else:
                oNumEventMC+=iNumEventMC


            if 'iFile' not in locals():
                iFile=0
            while 1:
                oUid=oData['uid'].unique()
                oUid.sort()

                if len(oUid)<self.numEventPerFile:
                    break
                else:
                    iUid=oUid[:self.numEventPerFile]
                    rData=oData[oData['uid']<=iUid[-1]]
                    oData=oData[oData['uid']>iUid[-1]]

                    iName=iChannelCSV+'{:0>7d}_{:0>7d}.csv'.format(iFile*self.numEventPerFile,(iFile+1)*self.numEventPerFile)
                    rData.to_csv(iName)

                    iFile+=1

                    if '_data' not in locals():
                        _data=rData




        iName=iChannelCSV+'{:0>7d}_{:0>7d}.csv'.format(iFile*self.numEventPerFile,iFile*self.numEventPerFile+len(oUid))
        oData.to_csv(iName)
        oNumEventMC_PID=iFile*self.numEventPerFile+len(oUid)

        if '_data' not in locals():
            _data=oData



        # If the channel is finished to process, then write it into runlog.
        with open(runLog,'w') as f:
            f.writelines('{:<20s}   :    {:>8d} \n{:<20s}   :    {:>8d} \nnumEventMC_PID/numEventMC   :     {:>8f}\n'.format('numEventMC_PID',oNumEventMC_PID,'numEventMC',oNumEventMC,oNumEventMC_PID/oNumEventMC))
            f.writelines('\n')
            f.writelines('{:<4s} {:<20s}  {:>12s} {:>12s} {:>12s} {:>12s} {:>4s}\n'.format('id','branch','mean','std','min','max','std=0'))
            [f.writelines('{:<4d} {:<20s} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} {:>4d}\n'.format(x,iBranch[x],_data[iBranch[x]].mean(),_data[iBranch[x]].std(),_data[iBranch[x]].min(),_data[iBranch[x]].max(),_data[iBranch[x]].std()==0.)) for x in range(len(iBranch))]

            f.writelines('\nfinished\n')   #     return

    def RootRead_Process_Write_Channels(self):       # return
        p=[]
        for iChannel in self.channels:
            p.append(Process(target=self.RootRead_Process_Write_Channel, args=(iChannel,)))

        rP=[]
        for iP in tqdm(range(len(p))):
            p[iP].start()
            rP.append(p[iP])

            # print(len(rP))
            while len(rP)>=self.numProcess:
                iBreak=0

                for irP in rP:
                    if not irP.is_alive():
                        rP.remove(irP)
                        iBreak=1
                        break
                if iBreak:
                    break

                time.sleep(0.1)

    def Run(self):
        self.RootRead_Process_Write_Channels()

    def Check(self):
        channelsChecked=[]
        channelsNotChecked=[]
        for iChannel in self.channels:
            iChannelCSV=self.homeCSV+iChannel+'/'
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


        checkLog=self.homeCSV+'checkLog'
        with open(checkLog,'w') as f:
            f.writelines('Check failed: \n')
            [f.writelines(x+'\n') for x in channelsNotChecked]

            f.writelines('\n'+'-'*40+'\n')
            f.writelines('Check successfully: \n')
            [f.writelines(x+'\n') for x in channelsChecked]

            f.writelines('\n'*3+'*'*40+'\n')
            f.writelines('\n'+'*'*5+' '*5+'    LIST  FORMAT    '+' '*5+'*'*5+'\n')
            f.writelines('\n'+'*'*40+'\n')
            f.writelines('Check failed: \n[')
            [f.writelines(x+', ') for x in channelsNotChecked]
            f.writelines(']\n'+'-'*40+'\n')
            f.writelines('Check successfully: \n[')
            [f.writelines(x+',') for x in channelsChecked]
            f.writelines(']')


        self.channelsChecked, self.channelsNotChecked=channelsChecked, channelsNotChecked
        self.flagChecked=len(channelsNotChecked)==0
        # return channelsChecked, channelsNotChecked,self.flagChecked
        return self.flagChecked

    def DelFolderChannelsNotChecked(self):
        for iChannel in self.channelsNotChecked:
            iChannelDel = self.homeCSV+iChannel

            shutil.rmtree(iChannelDel,ignore_errors=True)

    def SetChannel(self,channels):
        if isinstance(channels,list):
            self.channels=channels
            self.channels.sort()
        else:
            self.channels=[channels]

    def ReRun(self):
        self.Check()
        if self.flagChecked:
            return self.flagChecked
        self.DelFolderChannelsNotChecked()
        self.channelsBK=self.channels
        self.SetChannel(self.channelsNotChecked)
        self.Run()
        self.channels=self.channelsBK
        self.Check()
        return self.flagChecked

    def Check2Finish(self,timeSecondDelta=60,timeSecondTotal=7200,label=None):
        if label is None:
            label=type(self).__name__

        self.Check()
        if self.flagChecked:
            print(label+' : ', self.flagChecked, '  (Successful)')
            return self.flagChecked

        numCounter=timeSecondTotal/timeSecondDelta
        iCounter=-1
        while True:
            iCounter+=1
            if iCounter>=numCounter:
                assert self.flagChecked,label+' Failed, please try larger timeSecondTotal'
                print(label+' : ', self.flagChecked, '  (Failed)')
                return self.flagChecked

            print('Running  %s  x  %d  @ %s'%(label, iCounter,self.channelsNotChecked[0]))

            time.sleep(timeSecondDelta)

            self.Check()
            if self.flagChecked:
                print(label+' : ', self.flagChecked, '  (Successful)')
                return self.flagChecked








if __name__=='__main__':
    homeRoot='/home/i/iWork/data/root'
    homeCSV='/home/i/iWork/data/csv'
    # channels=['T09_T05_55_D0', 'T09_T06_55_Dch', 'T09_T07_55_Ds', 'T09_T08_55_Lam', 'T09_T09_55_Lamc']
    channels=None
    numEventPerFile=2048
    numProcess=10
    oRoot2CSV=Root2CSV(homeRoot=homeRoot,
                       homeCSV=homeCSV,
                       channels=channels,
                       numEventPerFile=numEventPerFile,
                       numProcess=numProcess)


    # oRoot2CSV.Run()      # First run
    oRoot2CSV.Check()
    oRoot2CSV.ReRun()






#
