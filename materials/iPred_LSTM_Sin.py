#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author  : P. Jiang



import numpy as np



import torch
from torch.utils.data import  Dataset
import  matplotlib.pyplot as plt

class DataSetSin(Dataset):
    def __init__(self,seqIn=10,seqOut=13,train_test='train'):
        super(DataSetSin,self).__init__()
        if train_test=='train':
            self.data=(np.sin(np.arange(1e6*50)*0.1)).reshape((-1,1,50))
        if train_test=='test':
            self.data=(np.sin(np.arange(1e6*50,1e6*50+10000)*0.1)).reshape((-1,1,50))


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        data=torch.from_numpy(self.data[idx,:,:seqIn]).float()
        label=torch.from_numpy(self.data[idx,:,seqIn:seqIn+seqOut]).float()
        return (data,label)




class DataSet(Dataset):
    def __init__(self,oData):
        super(DataSet,self).__init__()
        data=oData


from torch import nn
from torch.autograd import Variable
class PredLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(PredLSTM, self).__init__()


        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        print('0 0 0:   ',x.shape)
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        print('1 1 1 :   ',x.shape)
        return x






###################################

seqIn=10
seqOut=40
hiddenSize=20

batchSize=32

numEpoch=10


oDataSetSinTrain=DataSetSin(seqIn=seqIn,seqOut=seqOut,train_test='train')
oDataSetSinTest=DataSetSin(seqIn=seqIn,seqOut=seqOut,train_test='test')





iDataSetTrain=oDataSetSinTrain
iDataSetTest=oDataSetSinTest

loaderTrain=torch.utils.data.DataLoader(iDataSetTrain, batch_size=batchSize,shuffle=True, num_workers=6)
loaderTest=torch.utils.data.DataLoader(iDataSetTest, batch_size=batchSize,shuffle=False, num_workers=6)


iNet = PredLSTM(seqIn, hiddenSize,seqOut)
iNet=iNet.cuda()


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(iNet.parameters(), lr=1e-2)


for epoch in range(numEpoch):
    # print(epoch)

    running_loss = 0.0
    for i, data in enumerate(loaderTrain, 0):
        iNet.train()

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(inputs.shape, labels.shape)
        inputs=inputs .cuda()
        labels=labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = iNet(inputs)

        # print(outputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        numShow=200
        if i % numShow == 0:    # print every 2000 mini-batches
            iRunning_loss=running_loss/numShow
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, iRunning_loss))

            running_loss = 0.0


            iNet.eval()

            dataIterTest=iter(loaderTest)
            data=dataIterTest.next()


            inputs, labels = data
            inputs=inputs .cuda()
            labels=labels.cuda()

            # print(inputs.shape,labels.shape)

            outputs = iNet(inputs)

            inputs=inputs.cpu()
            labels=labels.cpu()
            outputs=outputs.detach().cpu()

            plt.figure('Test')
            plt.clf()
            plt.plot(np.arange(inputs.shape[2]),inputs[0,0,:],'b.')
            plt.plot(inputs.shape[2]+np.arange(labels.shape[2]),labels[0,0,:],'go')
            plt.plot(inputs.shape[2]+np.arange(outputs.shape[2]),outputs[0,0,:],'r.')
            plt.grid()

            plt.pause(0.1)


print('Finished Training')














#
