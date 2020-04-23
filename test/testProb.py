import numpy as np

x0=np.array([1.5,2,2.5])

x=x0.copy()
p=0
for i in range(10000):
    print(x,p)
    x_1=1/x
    p=x_1/(np.sum(x_1)*len(x))

    pRand=np.random.random((len(x)))

    for j in range(len(x)):
        if pRand[j]<p[j]:
            x[j]+=x0[j]


#
