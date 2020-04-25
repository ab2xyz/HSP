#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
class Channel():
        def __init__(self,homeCSV=None,channels=None):
            self.homeCSV=homeCSV
            if self.homeCSV.strip()[-1]!='/':
                self.homeCSV=self.homeCSV+'/'

            channels=self.Channel(home=self.homeCSV,channels=channels)
            self.channels=channels


        def Channel(self,home,channels):
            # # channels
            # if channels is None:
            #     channels=[x for x in os.listdir(home) if os.path.isdir(home+x)]
            #
            # else:
            #     if isinstance(channels,list):
            #         channels=channels
            #         channels.sort()
            #     else:
            #         channels=[channels]


            if isinstance(channels,list):
                channels=channels

            else:
                if channels is None:
                    channels=[x for x in os.listdir(home) if os.path.isdir(home+x)]
                else:

                    channels=[x for x in os.listdir(home) if os.path.isdir(home+x) and channels in x]



            channels.sort()

            return channels


        def GetChannel(self,):
            print(self.channels)
            return self.channels


if __name__=='__main__':
    homeCSV='/home/i/iWork/data/csv'
    # channels=['T09_T05_55_D0', 'T09_T06_55_Dch', 'T09_T07_55_Ds', 'T09_T08_55_Lam', 'T09_T09_55_Lamc']
    # channels=None
    channels='45'
    oChannel=Channel(homeCSV=homeCSV,channels=channels)
    oChannel.GetChannel()



#
