#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

class RootSetSplit():
    def __init__(self,homeRoot, homeCSV, channels=None, train=0.7,test=0.3):
        self.homeRoot=homeRoot if homeRoot.strip()[-1]=='/' else homeRoot+'/'
        self.homeCSV=homeRoot if homeCSV.strip()[-1]=='/' else homeCSV+'/'

        self.channels=self.Channel(home=self.homeRoot,channels=channels)

        print()
        print(self.channels)

        self.GetSymbolData()


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
        self.symbolData=[]
        for iChannel in self.channels:

            self.symbolData.append(iChannel[iChannel.find('_')+1:])
        self.symbolData=list(set(self.symbolData))
        self.symbolData.sort()

        print(self.symbolData)

    # def ChannelDPM(self):
    #     self.symbolData
    #     for iChannle in





if __name__=='__main__':

    homeRoot='/home/i/iWork/data/root'
    homeCSV='/home/i/IGSI/data/csv'
    channels=None

    train=0.7
    test=0.3

    oRootSetSplit=RootSetSplit(homeRoot=homeRoot, homeCSV=homeCSV, channels=channels,train=train, test=test)
    pass


#
