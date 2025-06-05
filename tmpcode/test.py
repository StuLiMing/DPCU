import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Range(nn.Module):
    def __init__(self,in_ch,out_ch,kernel=(3,5), padding=(1,2)):
        super().__init__()
        self.x=padding[1]
        self.y=padding[0]
        self.conv=nn.Conv2d(in_ch,out_ch,kernel)

    def forward(self, x):
        x=F.pad(x, (0, 0,self.y,self.y), mode='reflect')
        x=F.pad(x, (self.x, self.x,0,0), mode='circular')
        return self.conv(x)

