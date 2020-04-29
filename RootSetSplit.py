#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os,shutil

class RootSetSplit():
    def __init__(self,homeRoot, homeData, channels=None, train=0.7):
        '''
        Split root set to train set and test set.
        homeRoot is the home of root files.
        homeData is the folder to put all data including the softlinks of roots and the csvs.
        channels is the channels to precess.
        train is the ratio of train set.

        '''
        self.homeRoot=homeRoot if homeRoot.strip()[-1]=='/' else homeRoot+'/'
        self.homeData=homeRoot if homeData.strip()[-1]=='/' else homeData+'/'
        os.makedirs(self.homeData,exist_ok=True)

        self.channels=self.Channel(home=self.homeRoot,channels=channels)

        self.train=train


        self.GetSymbolData()
        self.GetSymbolDataChannel()
        # self.GetSymbolDataTriggerLine()
        # self.GetSymbolDataFile()
        self.GetSymbolDataFileEvent()
        self.RootLn()


    def Channel(self,home,channels):

        if isinstance(channels,list):
            channels=channels

        else:
            if channels is None:
                channels=[x for x in os.listdir(home) if os.path.isdir(home+x)]
            else:
                channels=[x for x in os.listdir(home) if os.path.isdir(home+x) and channels in x]

        channels.sort()

        return channels



    def GetSymbolData(self):
        '''
        list of symbol of data   (data is data in [data, trigger])
        self.symbolData=['DPM_24', 'DPM_38', 'DPM_45', ... 'T00_38_ee', 'T00_45_ee',  ... 'T09_55_Lamc']
        len: 34
        '''
        self.symbolData=[]
        for iChannel in self.channels:

            self.symbolData.append(iChannel[iChannel.find('_')+1:])
        self.symbolData=list(set(self.symbolData))
        self.symbolData.sort()

    def GetSymbolDataChannel(self):
        '''
        dict { symbol of data: channel}     (data is data in [data, trigger])

        self.symbolDataChannel={
        DPM_24  :  ['T00_DPM_24', 'T01_DPM_24', 'T08_DPM_24']

        DPM_38  :  ['T00_DPM_38', 'T01_DPM_38',  ... 'T08_DPM_38']
        .
        .
        .
        T09_55_Lamc  :  ['T00_T09_55_Lamc', 'T01_T09_55_Lamc', ... 'T09_T09_55_Lamc']
        }
        '''

        self.symbolDataChannel={}
        for iSymbolData in self.symbolData:
            self.symbolDataChannel[iSymbolData]=[]
            for iChannel in self.channels:
                if iSymbolData in iChannel:
                    self.symbolDataChannel[iSymbolData].append(iChannel)
            self.symbolDataChannel[iSymbolData].sort()



    def LogGetNumEventMC(self,iLog):  #     return numEventMC
        '''
        Read log to get number of event MC.

        return numEventMC
        '''

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


    def GetSymbolDataFileEvent(self):
        '''
        Get the symbolFileTrain and symbolFileTest, and write them into file rootSplit.log.
        No return.

        '''
        self.symbolDataFileEvent={}
        self.symbolDataEvent={}
        for iSymbolData in self.symbolDataChannel:
            iChannel=self.symbolDataChannel[iSymbolData][0]
            self.symbolDataFileEvent[iSymbolData]={}
            self.symbolDataEvent[iSymbolData]=0
            logs=[x.replace('.root','.log') for x in os.listdir(self.homeRoot+iChannel) if  x[-5:]=='.root']
            for iLog in logs:
                symFile=iLog[iLog.find('_')+1:-4]
                numEventMC=self.LogGetNumEventMC(self.homeRoot+iChannel+'/'+iLog)
                self.symbolDataFileEvent[iSymbolData][symFile]=numEventMC
                self.symbolDataEvent[iSymbolData]+=numEventMC


        self.symbolFileTrain={}
        self.symbolFileTest={}

        for iSymbolData in self.symbolDataFileEvent:
            cutTrain=self.train*self.symbolDataEvent[iSymbolData]

            counterTrain=0
            self.symbolFileTrain[iSymbolData]=[]
            self.symbolFileTest[iSymbolData]=[]
            for isymbolFile in self.symbolDataFileEvent[iSymbolData]:
                counterTrain+=self.symbolDataFileEvent[iSymbolData][isymbolFile]
                if counterTrain<=cutTrain:
                    self.symbolFileTrain[iSymbolData].append(isymbolFile)
                else:
                    self.symbolFileTest[iSymbolData].append(isymbolFile)



        rootSplitLog=self.homeData+'rootSplit.log'
        with open (rootSplitLog,'w') as f:
            f.close()
        for iSymbolData in self.symbolDataFileEvent:
            with open (rootSplitLog,'a') as f:
                f.writelines('symbolData  :    %s\n'%iSymbolData)
                f.writelines('train :\n')
                [f.writelines('%s    '%x) for x in  self.symbolFileTrain[iSymbolData]]
                f.writelines('\n'+'-'*80+'\ntest :\n')
                [f.writelines('%s    '%x) for x in  self.symbolFileTest[iSymbolData]]
                f.writelines('\n'+'='*80+'\n')


    def GetHomeCSV(self):
        return self.folderCSVTrain,self.folderCSVTest

    def RootLn(self):
        '''
        Soft link from homeRoot to homeData
        '''
        self.folderCSV=self.homeData+'csv/'
        self.folderCSVTrain=self.folderCSV+'train/'
        self.folderCSVTest=self.folderCSV+'test/'
        os.makedirs(self.folderCSVTrain,exist_ok=True)
        os.makedirs(self.folderCSVTest,exist_ok=True)

        self.folderRoot=self.homeData+'root/'
        self.folderTrain=self.folderRoot+'train/'
        self.folderTest=self.folderRoot+'test/'

        shutil.rmtree(self.folderTrain,ignore_errors=True)
        shutil.rmtree(self.folderTest,ignore_errors=True)


        os.makedirs(self.folderTrain,exist_ok=True)
        os.makedirs(self.folderTest,exist_ok=True)

        for iSymbolData in self.symbolDataChannel:
            for iChannel in self.symbolDataChannel[iSymbolData]:
                ifolderTrain=self.folderTrain+iChannel+'/'
                ifolderTest=self.folderTest+iChannel+'/'

                os.makedirs(ifolderTrain,exist_ok=True)
                os.makedirs(ifolderTest,exist_ok=True)

                symFiles=[x[x.find('_')+1:-5] for x in os.listdir(self.homeRoot+iChannel) if x[-5:]=='.root']
                iTrigger=iChannel[:iChannel.find('_')+1]

                for iFile in symFiles:
                    if iFile in self.symbolFileTrain[iSymbolData]:
                        oFolder=ifolderTrain

                    if iFile in self.symbolFileTest[iSymbolData]:
                        oFolder=ifolderTest

                    iRoot=self.homeRoot+iChannel+'/'+iTrigger+iFile+'.root'
                    iLog=self.homeRoot+iChannel+'/'+iTrigger+iFile+'.log'

                    oRoot=oFolder+iTrigger+iFile+'.root'
                    oLog=oFolder+iTrigger+iFile+'.log'


                    os.symlink(iRoot,oRoot)
                    os.symlink(iLog,oLog)






if __name__=='__main__':

    homeRoot='/home/i/iWork/data/root'
    homeData='/home/i/IGSI/data/data'
    channels=['T00_DPM_45','T00_T08_45_Lam']
    channels='45'
    channels=None

    train=0.7

    oRootSetSplit=RootSetSplit(homeRoot=homeRoot, homeData=homeData, channels=channels,train=train)


#
