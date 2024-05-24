import torch
import torch.nn as nn
import numpy as np

class Tower(nn.Module):
    def __init__(self,field_dims,embedding_dims,hidden_nums,output_nums):
        super(Tower,self).__init__()
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

class doubleTower(nn.Module):
    def __init__(self,users_dims,item_dims):
        super(doubleTower, self).__init__()
        self.userTower = Tower(users_dims,32,64,3)
        self.itemTower = Tower(item_dims,16,32,3)


    def forward(self,user_input,item_input):
        Qu = self.userTower(user_input)
        Qi = self.itemTower(item_input)
        # print("shape before cos",Qu.shape,Qi.shape)
        cos_ui = nn.functional.cosine_similarity(Qu, Qi)
        value = (cos_ui+1)/2
        return value



