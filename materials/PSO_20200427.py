import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def FuncCuts(x, **kwargs):
    a=kwargs['a']
    b=kwargs['b']

    y0=a*x[0]**2+b**2*x[1]**3
    y1=a**2*x[0]**3-b*x[1]**2

    return y0,y1


def PSOND(Func,lb,ub, swarmsize=500,maxiter=100, memorysize=1000,pltShow=True, *args,  **kwargs):
    lb,ub=np.array(lb),np.array(ub)

    mb=(lb+ub)/2

    numB=len(mb)

    target=Func(mb, *args, **kwargs)

    numTarget = len(target)


    lwcc=np.ones((numTarget+3))*0.001
    uwcc=np.ones((numTarget+3))*3.

    lb=np.r_[lb,lwcc]
    ub=np.r_[ub,uwcc]

    mb=(lb+ub)/2
    numB=len(mb)


    for iIter in tqdm(range(maxiter)):
        # print(iIter)
        if iIter==0:

            popu=np.random.random((swarmsize//1,numB))*(ub-lb)+lb
            vPopu=np.random.random((swarmsize//1,numB))*(ub-lb)*0.01


            popuMemo=np.zeros((memorysize//1+swarmsize//1,numB))
            popuMemoMarks=np.ones((memorysize//1+swarmsize//1,numTarget))*1e15


            popuLocal=np.zeros((swarmsize//1,numB,numTarget))
            popuLocalMarks=np.ones((swarmsize//1,numTarget))*1e15

            numMemo=0

        else:

            idx0=np.random.choice(range(numMemo),size=swarmsize)
            idx1=np.random.choice(range(numMemo),size=swarmsize)
            idx2=np.random.choice(range(numMemo),size=swarmsize)
            idx3=np.random.choice(range(numMemo),size=swarmsize)

            r=[]
            for iR in range(numTarget+2):
                r.append(np.random.random()/(numTarget+2))


            wcc=[]
            nRepeat=vPopu.shape[1]
            for iWcc in range(-(numTarget+3),0):
                wcc.append(popu[:,iWcc][:,np.newaxis].repeat(nRepeat,axis=1))

            idx1=np.random.choice(range(numMemo),size=swarmsize)
            idx2=np.random.choice(range(numMemo),size=swarmsize)

            vPopu=wcc[-(numTarget+3)]*vPopu+r[0]*wcc[-(numTarget+2)]*(popuMemo[idx1,:]-popu)+r[1]*wcc[-(numTarget+1)]*(popuMemo[idx2,:]-popu)

            for iNumTarget in range(numTarget):
                vPopu+=r[iNumTarget+2]*wcc[-(numTarget-iNumTarget)]*(popuLocal[:,:,iNumTarget]-popu)


            popu=popu+vPopu


            for idxB in range(numB):
                popu[popu[:,idxB]<lb[idxB],idxB]=lb[idxB]
                popu[popu[:,idxB]>ub[idxB],idxB]=ub[idxB]


        for iSwarm in range(swarmsize//1):

            target=Func(popu[iSwarm,:-3], *args, **kwargs)

            tar=np.array(target)
            tarFlag=((tar>popuMemoMarks[:numMemo]).all(axis=1)).any()

            if tarFlag:
                # Be dominated
                continue

            tarFlag=~(tar<popuMemoMarks[:numMemo]).all(axis=1)

            popuMemoMarksBK=popuMemoMarks[:numMemo][tarFlag,:].copy()
            popuMemoBK=popuMemo[:numMemo][tarFlag,:].copy()

            numMemo=popuMemoMarksBK.shape[0]+1
            popuMemoMarks[:numMemo-1,:]=popuMemoMarksBK.copy()
            popuMemo[:numMemo-1,:]=popuMemoBK.copy()


            popuMemoMarks[numMemo-1,:]=tar
            popuMemo[numMemo-1,:]=popu[iSwarm,:].copy()


            ##

            for iTarget in range(numTarget):
                if tar[iTarget]<=popuLocalMarks[iSwarm,iTarget]:
                    popuLocalMarks[iSwarm,iTarget]=tar[iTarget]
                    popuLocal[iSwarm,:,iTarget]=popu[iSwarm,:].copy()


        if numMemo>memorysize:     #delete dense particles
            num=numMemo-memorysize
            for iDel in range(num):
                idxRand=np.random.choice(range(numMemo),3)

                iR=[]
                for iRand in idxRand:
                    popuMemoMarksNorm=(popuMemoMarks-popuMemoMarks[:numMemo,:].min(axis=0))/popuMemoMarks[:numMemo,:].max(axis=0)

                    iMark=popuMemoMarksNorm[iRand,:]
                    iMarkDiff=popuMemoMarksNorm-iMark


                    iNorm=(iMarkDiff**2).sum(axis=1)
                    iNorm.sort()
                    iR.append(iNorm[:10].sum())

                idx=idxRand[np.argmin(iR)]

                popuMemoMarks[idx,:]=popuMemoMarks[numMemo-1,:]
                popuMemo[idx,:]=popuMemo[numMemo-1,:]

                numMemo-=1





        if pltShow:

            plt.figure('Fit')
            plt.clf()
            plt.plot(popuMemoMarks[:numMemo,0],popuMemoMarks[:numMemo,1],'.')


            plt.grid()
            plt.title(numMemo)
            plt.pause(0.01)




    if 'popuRecommand' not in locals():

        popuMemoMarksNorm=popuMemoMarks[:numMemo,:]-popuMemoMarks[:numMemo,:].min()
        popuMemoMarksNorm=popuMemoMarksNorm/popuMemoMarks[:numMemo,:].max()

        popuRecommand=popuMemo[np.argmin(np.linalg.norm(popuMemoMarksNorm,axis=1,ord=2)),:-(numTarget+3)]


    popuMemo=popuMemo[:numMemo,:-(numTarget+3)]
    popuMemoMarks=popuMemoMarks[:numMemo,:]
    popuRecommand=popuRecommand


    print(popuMemo.shape,popuMemoMarks.shape)
    return popuMemo, popuMemoMarks,popuRecommand




if __name__ =='__main__':

    # FuncCuts(self,x, **kwargs)
    lb=[-10,-10]
    ub=[30,30]
    popuMemo, popuMemoMarks,popuRecommand=PSOND(FuncCuts,lb,ub, swarmsize=500,maxiter=100, memorysize=1000,pltShow=True,  a=2,b=-3)



    pass







    plt.show()

#
