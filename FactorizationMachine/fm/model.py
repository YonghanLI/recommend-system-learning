import torch
import torch.nn as nn
import numpy as np


class FL(nn.Module):
    def __init__(self,field_dims,output_dim):
        super().__init__()
        self.FL_linear = nn.Embedding(sum(field_dims),output_dim)
        self.FL_bias = nn.Parameter(torch.zeros(output_dim))
        self.offsets = np.array((0, *np.cumsum(field_dims[:-1])), dtype=np.long)

    def forward(self,x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = torch.sum(self.FL_linear(x),dim=1)+self.FL_bias
        return x

class FE(nn.Module):
    def __init__(self,field_dims,embedding_dim):
        self.FE_ = nn.Embedding(sum(field_dims),embedding_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims[:-1])), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.FE_.weight.data)

    def forward(self,x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = self.FE_(x)
        return x

class FMmodel(torch.nn.Module):
    def __init__(self,field_dims,output_dim,embedding_dim):
        super().__init__()
        # print("field_dims in Net",field_dims)
        self.FL=nn.Embedding(sum(field_dims),output_dim)
        self.FL_bias=torch.nn.Parameter(torch.zeros(output_dim))
        self.offsets=np.array((0,*np.cumsum(field_dims[:-1])),dtype=int)
        self.offsets[-1]=self.offsets[-3]
        self.offsets[-2]=self.offsets[-3]
        # print("offsets",self.offsets)
        self.FE=torch.nn.Embedding(sum(field_dims),embedding_dim)
        torch.nn.init.xavier_uniform_(self.FE.weight.data)

        self.fc = nn.Linear(embedding_dim,output_dim)

    def forward(self,x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # print("x in",x.shape)
        x0=torch.sum(self.FL(x),dim=1)+self.FL_bias
        # print("Bias shape",self.FL_bias.shape)
        x1=self.FE(x)
        y=0.5*(torch.sum(x1,dim=1)**2 - torch.sum(x1**2,dim=1))
        # print('x0 shape',x0.shape,"y shape",y.shape)
        # z=x0+self.fc(y)
        # print("x0 shape",x0.shape,"y",y.shape)
        z=x0+torch.sum(y,dim=1, keepdim=True)
        # print("z shape",z.shape)
        output=torch.sigmoid(z.squeeze(1))

        return output













