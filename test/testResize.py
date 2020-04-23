import torch

x1=torch.arange(1.,7.)
x2=torch.zeros(9,dtype=torch.float)
x2[:len(x1)]=x1
x3=x2.resize_(3,3)

x=torch.arange(1.,7.).resize_(2,5)


print(x2)
