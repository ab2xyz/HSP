import pandas as pd




def WriteEffi2CSV(homeRes,codeSave):

    homeRes=homeRes if homeRes.strip()[-1]=='/' else homeRes+'/'
    fEffiLog=homeRes+codeSave+'/effi'+codeSave+'.log'



    with open(fEffiLog,'r') as f:
        contentEffi=f.readlines()


    listTrigger=[]
    listData=[]
    listEffiReconstruction=[]
    listEffiTriggerReconstruction=[]
    listEffiTriggerRaw=[]

    for iEffi in contentEffi:
        iEffiSplit=iEffi.split('\n')[0].strip()
        for i in range(len(iEffiSplit)//2):
            iEffiSplit=iEffiSplit.replace('  ',' ')

        iEffiSplit=iEffiSplit.split(' ')


        iChannle,iEffiReconstruction,iEffiTriggerReconstruction, iEffiTriggerRaw=iEffiSplit

        idxTrigger=iChannle.find('_')
        if idxTrigger==-1:
            continue
        iTrigger=iChannle[:idxTrigger]
        iData=iChannle[idxTrigger+1:idxTrigger+1+iChannle[idxTrigger+1:].find('_')]

        listTrigger.append(iTrigger)
        listData.append(iData)
        listEffiReconstruction.append(iEffiReconstruction)
        listEffiTriggerReconstruction.append(iEffiTriggerReconstruction)
        listEffiTriggerRaw.append(iEffiTriggerRaw)

    listTriggerUnique=list(set(listTrigger))
    listTriggerUnique.sort()
    listDataUnique=list(set(listData))
    listDataUnique.sort()


    fEffiReconstructionCSV=homeRes+codeSave+'/effiReconstruction'+codeSave+'.csv'
    iEffiReconstructionPd=pd.DataFrame(columns=listTriggerUnique,index=listDataUnique)

    fEffiTriggerReconstructionCSV=homeRes+codeSave+'/effiTriggerReconstruction'+codeSave+'.csv'
    iEffiTriggerReconstructionPd=pd.DataFrame(columns=listTriggerUnique,index=listDataUnique)

    fEffiTriggerRawCSV=homeRes+codeSave+'/effiTriggerRaw'+codeSave+'.csv'
    iEffiTriggerRawPd=pd.DataFrame(columns=listTriggerUnique,index=listDataUnique)



    for i in range(len(listTrigger)):
        iEffiReconstructionPd.loc[listData[i],listTrigger[i]]=listEffiReconstruction[i]
        iEffiTriggerReconstructionPd.loc[listData[i],listTrigger[i]]=listEffiTriggerReconstruction[i]
        iEffiTriggerRawPd.loc[listData[i],listTrigger[i]]=listEffiTriggerRaw[i]


    iEffiReconstructionPd.to_csv(fEffiReconstructionCSV)
    iEffiTriggerReconstructionPd.to_csv(fEffiTriggerReconstructionCSV)
    iEffiTriggerRawPd.to_csv(fEffiTriggerRawCSV)






if __name__=='__main__':
    homeRes='/home/i/IGSI/data/res'
    codeSave='DNNTestBN'
    WriteEffi2CSV(homeRes=homeRes,codeSave=codeSave)









#
