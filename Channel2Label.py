#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Channel import  Channel

class Channel2Label(Channel):
    def __init__(self,homeCSV,channels=None):
        super(Channel2Label,self).__init__(homeCSV,channels)

        self.Data2Label()
        self.Channel2Label()


    def GetChannel2Label(self,iChannel=None):
        if iChannel is None:
            return self.channel2Label
        else:
            return self.channel2Label[iChannel]

    def GetNumClass(self):
        return self.numClasses


    def Data2Label(self):
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


        self.data2Label=dict(zip(self.classes,labels))

        self.numClasses=len(labels)


    def Channel2Label(self):
        channel2Label={}
        for iChannel in self.channels:
            channel2Label[iChannel]=self.Channel2Data4Label(iChannel)
        self.channel2Label=channel2Label

        # [print(x,y) for x,y in self.channel2Label.items()]


    def Channel2Data4Label(self,iChannel):
        iChannelList=iChannel.split('_')[:3:]
        if iChannelList[0] == iChannelList[1]:
            iClass='_'.join(iChannelList[0:3:2])
        else:
            iClass='DPM_'+iChannelList[2]

        return self.data2Label[iClass]


if __name__=='__main__':
    homeCSV='/home/i/IGSI/data/data/csv/train'
    channels=None
    oChannel2Label=Channel2Label(homeCSV=homeCSV,channels=channels)



#
