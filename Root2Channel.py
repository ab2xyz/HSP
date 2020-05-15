#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import shutil
import os


class Root2Channel():
    def __init__(self,homeRoot):
        self.homeRoot=homeRoot if homeRoot.strip()[-1]=='/' else homeRoot+'/'

        fList=os.listdir(self.homeRoot)
        rootList=[ x for x in fList if x[-5:]=='.root']
        logList=[ x for x in fList if x[-4:]=='.log']

        for iRoot in rootList:
            iLog=iRoot[:-5]+'.log'
            if not iLog in logList:
                continue


            iChannel=iRoot[:iRoot.find('_ana_')]
            iFolder=self.homeRoot+iChannel
            os.makedirs(iFolder,exist_ok=True)

            fRoot=self.homeRoot+iRoot
            fLog=self.homeRoot+iLog

            shutil.move(fRoot,iFolder)
            shutil.move(fLog,iFolder)




if __name__=='__main__':

    homeRoot='/home/i/IGSI/data/root'

    oRoot2Channel=Root2Channel(homeRoot=homeRoot)

    print('End')




#
