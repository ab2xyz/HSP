
#encoding:utf-8
import torch
import torch.nn as nn
import numpy as np

#num_features - num_features from an expected input of size:batch_size*num_features*height*width
#eps:default:1e-5 (公式中为数值稳定性加到分母上的值)
#momentum:动量参数，用于running_mean and running_var计算的值，default：0.1

m=nn.BatchNorm2d(2,affine=True) #affine参数设为True表示weight和bias将被使用
input=torch.randn(2,2,3,4)
output=m(input)

print(input)
print(m.weight)
print(m.bias)
print(output)
print(output.size())




def bn_from_scratch(inputs):
    """do batch normalization when train without scale and shift
    :param: [batch_size, channels, height, width]
    """
    # compute mean over channels
    mean = torch.mean(inputs, dim=(0, 2, 3))
    # reshape for broadcast
    mean = mean.view(1, inputs.size(1), 1, 1)
    # compute std using biased way, plus epsilon for stability
    std = torch.sqrt(torch.var(inputs, dim=(0, 2 , 3), unbiased=False) + 1e-5)
    std = std.view(1, inputs.size(1), 1, 1)
    invstd = 1/std
    # Core steps, do scale and shift here if wanted
    test_outputs = (inputs-mean)*invstd
    return test_outputs




test_outputs=bn_from_scratch(inputs=input)

print(test_outputs)



print('/'*80)
