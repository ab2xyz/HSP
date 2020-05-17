

import torch
import numpy as np

dataSig_Bkg=np.array([5,1,6,4,4,8,9,5,7,3,4])    # Origin values
uidBkg=np.array([2,3,3,4,4,5,5,2,2,3,3])     # origin keys
# idx=torch.range(0,uidBkg.shape[0]-1).long()




dataSig_BkgArgsort=np.argsort(dataSig_Bkg)    # sort : Origin values   -  from larger to less
dataSig_BkgSort=dataSig_Bkg[dataSig_BkgArgsort]    # dataSig_Bkg Sort
uidBkgSort=uidBkg[dataSig_BkgArgsort]     #  keys sorted with Origin-value-sorting...

dictUid_DataSig=dict(zip(uidBkgSort,dataSig_BkgSort))


print(dataSig_BkgArgsort)
print(dataSig_BkgSort)
print(uidBkgSort)

[print(x,y) for x,y in dictUid_DataSig.items()]


print('*'*10)

xTest=[2,3,3,4,4,5,5,2,2,3,3]
yTest=[5,1,6,4,4,8,9,5,7,3,4]


iDict=dict(zip(xTest,yTest))
[print(x,y) for x,y in iDict.items()]


#
# # idxSort=idx[dataSig_BkgSort]
#
# #
# uidBkgSortUnique,uidBkgSortUniqueIdx=torch.unique_consecutive(uidBkgSort,return_inverse=True)     # Keys (keys sorted with Origin-value-sorting) unique
# #
# # dataSig_BkgTest=0
#
# print('-'*80)
# print(dataSig_Bkg)
# print(uidBkg)
# # print(idx)
#
# print('-'*80)
# print(dataSig_Bkg)
# print(dataSig_BkgSort)
# print(uidBkgSort)
# # print(idxSort)
# print(uidBkgSortUnique)
# print(uidBkgSortUniqueIdx)
#

# print(dataSig_BkgSort)
# print(uidBkgSort)
# print(uidBkgSortUnique)
# print(uidBkgSortUniqueIdx)

#
