import numpy as np

x=np.arange(30)
x_6_5=x.reshape((6,5))



print(x)
print(x_6_5)

x_1_6_5=x.reshape((1,6,5))
print(x_6_5)


###############

print('.'*80)

x=np.arange(60)
x_5_12=x.reshape((5,12))



print(x_5_12)

x_5_3_4=x_5_12.reshape(5,1,3,4)
print(x_5_3_4)


#
