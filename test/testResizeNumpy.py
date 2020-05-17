import numpy as np

x=np.arange(30)

y=np.resize(x,(5,6))

z=np.copy(y)
z[:,4:]=0
print(y)

print(z)


zz=np.resize(z,(5,2,3))

print(zz)


# yy=np.resize(y,(5,2,4))
#
# print('*'*20)
# print(y)
# print(yy)
