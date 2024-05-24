import torch
import torch.nn as nn
import numpy as np
from model import Tower, doubleTower
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from data_get import train_data
from sklearn.metrics import roc_auc_score


epoch_nums = 20
Batch_Size = 1024
Num_Workers = 8
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-3


dataset = train_data("./data_pre.txt")
# print(dataset.items[0])
train_length = int(len(dataset) * 0.8)
valid_length = int(len(dataset) * 0.1)
test_length = len(dataset) - train_length - valid_length
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_length, valid_length, test_length))

train_data_loader = DataLoader(train_dataset, batch_size=Batch_Size, num_workers=Num_Workers)
valid_data_loader = DataLoader(valid_dataset, batch_size=Batch_Size, num_workers=Num_Workers)
test_data_loader = DataLoader(test_dataset, batch_size=Batch_Size, num_workers=Num_Workers)
# print("dataset field dims",dataset.field_dims)

net = doubleTower(dataset.users_dims,dataset.items_dims).to(device)

lossfunc=torch.nn.BCELoss()
optimizer_v=torch.optim.Adam(net.parameters(),lr=learning_rate)

def train_process(net,optimizer,data_loader,lossfunc,device):
    print('train now')
    net.train()
    for users,items,target in tqdm(data_loader):
        users,items, target = users.to(device).long() ,items.to(device).long() , target.to(device).long()
        out = net(users,items)
        # print("target shape",target.shape,"out shape",out.shape)
        target = target.float()
        loss = lossfunc(out,target)
        # print(" Loss ",loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # tqdm.write('loss now: {}'.format(loss.item()))


def test(net,data_loader,device):
    print('test now')
    net.eval()
    targets, predicts = [],[]
    with torch.no_grad():
        for users, items, target in tqdm(data_loader):
            users, items, target = users.to(device).long(), items.to(device).long(), target.to(device).long()
            out = net(users,items)
            targets.extend(target.tolist())
            predicts.extend(out.tolist())
    result = roc_auc_score(targets,predicts)
    return result

if __name__ == '__main__':
    print('Working on', device)
    print("It's double tower")
    for i in tqdm(range(epoch_nums)):
        train_process(net,optimizer_v,train_data_loader,lossfunc,device)
        result = test(net,test_data_loader,device)
        print('AUC now:',result)

    last_result = test(net,test_data_loader,device)
    print('In the end :',last_result)















