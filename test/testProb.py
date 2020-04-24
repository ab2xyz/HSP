import numpy as np

x0=np.array([1,50,3])

x=np.zeros_like(x0)



prob=np.ones_like(x0)
for i in range(100):

    for i in range(len(x)):
        if prob[i]>0:
            x[i]+=x0[i]

    xMean=x.mean()


    for i in range(len(x)):
        if x[i]>xMean:
            prob[i]=0
        else:
            prob[i]=1

    print(x)  # ,print(xMean)













# x=x0.copy()
# p=0
# for i in range(10000):
#     print(x,p)
#     x_1=1/x
#     p=x_1/(np.sum(x_1)*len(x))
#
#     pRand=np.random.random((len(x)))
#
#     for j in range(len(x)):
#         if pRand[j]<p[j]:
#             x[j]+=x0[j]


#
