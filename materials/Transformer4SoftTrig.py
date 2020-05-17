import torch
from torch import  nn


class Transformer4SoftTrig(nn.Module):
    def __init__(self,d_model=256,nhead=2, numLayers=3,numCandidate=1,numClasses=2):
        super(Transformer4SoftTrig,self).__init__()
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder=nn.TransformerEncoder(self.encoderLayer, num_layers=numLayers)


        self.encoderLayer_0 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoderLayer_1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoderLayer_2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoderLayer_3 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoderLayer_4 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoderLayer_5 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        self.conv1 = nn.Conv1d(numCandidate, 1, 1, stride=1)


        self.fc=nn.Linear(d_model*numCandidate,numClasses)

    def forward(self, x):
        print('00  ' , x.shape)
        x=self.encoderLayer_0(x)

        res=x
        x=self.encoderLayer_1(x)
        res=res+x
        x=self.encoderLayer_2(x)
        res=res+x
        x=self.encoderLayer_3(x)
        res=res+x
        x=self.encoderLayer_4(x)
        res=res+x
        x=self.encoderLayer_5(x)
        res=res+x

        # print('11  ' , x.shape)
        x=res

        x=x.permute(1,0,2)
        # x=torch.tanh(self.conv1(x))
        # x=x.squeeze()
        print('000 ',x.shape)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x



if __name__=='__main__':
    modelTransformer4SoftTrig=Transformer4SoftTrig()
