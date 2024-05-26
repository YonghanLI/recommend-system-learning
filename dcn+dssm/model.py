import torch
import torch.nn as nn
import numpy as np

class DNN(nn.Module):
    def __init__(self,field_dims,embedding_dims,hidden_nums,output_nums):
        super(DNN,self).__init__()
        self.embedding = nn.Embedding(sum(field_dims),embedding_dims)
        self.fc = nn.Sequential(
            nn.Linear(len(field_dims)*embedding_dims,hidden_nums),
            nn.BatchNorm1d(hidden_nums),
            nn.ReLU(),
            nn.Linear(hidden_nums,hidden_nums),
            nn.BatchNorm1d(hidden_nums),
            nn.ReLU(),
            nn.Linear(hidden_nums, hidden_nums),
            nn.BatchNorm1d(hidden_nums),
            nn.ReLU(),
            nn.Linear(hidden_nums,output_nums),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.embedding(x)
        # print("x shape",x.shape)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x

class DCN(nn.Module):
    def __init__(self,cross_layers_num,field_dims,embedding_dims,output_nums):
        super(DCN,self).__init__()
        self.cross_layer_num = cross_layers_num
        self.embedding = nn.Embedding(sum(field_dims),embedding_dims)
        self.embedding_linear = nn.Linear(len(field_dims)*embedding_dims,output_nums)
        self.linear = nn.Linear(output_nums,output_nums)

    def forward(self,x):
        x = self.embedding(x)
        x = self.embedding_linear(x.view(x.shape[0],-1))
        x0 = x
        for _ in range(self.cross_layer_num):
            x =x0*self.linear(x) + x
        return x

class Tower(nn.Module):
    def __init__(self, field_dims,dnn_output_dims,dcn_output_dims):
        super(Tower,self).__init__()
        self.dnn = DNN(field_dims,32,64,dnn_output_dims)
        self.dcn = DCN(3,field_dims, 32, dcn_output_dims)

    def forward(self,x):
        x = torch.cat(
            [
            self.dnn(x),
            self.dcn(x)
            ],
            dim=-1
        )
        return x


class doubleTower(nn.Module):
    def __init__(self,users_dims,item_dims):
        super(doubleTower, self).__init__()
        self.userTower = Tower(users_dims,4,8)
        self.itemTower = Tower(item_dims,4,8)


    def forward(self,user_input,item_input):
        Qu = self.userTower(user_input)
        Qi = self.itemTower(item_input)
        # print("shape before cos",Qu.shape,Qi.shape)
        cos_ui = nn.functional.cosine_similarity(Qu, Qi)
        value = (cos_ui+1)/2
        return value



