import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd


class train_data(torch.utils.data.Dataset):
    def __init__(self,dataset_path):
        data_file = pd.read_csv(dataset_path, sep=",", engine="python", header="infer")
        new_columns = ["gender","age","occupation","user_id","genre","movie_id","score"]
        data_file = data_file.reindex(columns=new_columns)
        self.data = data_file.to_numpy()
        self.users = self.data[:,:-3].astype(int)
        self.items = self.data[:, -3:-1].astype(int)
        self.target = self.__preprocess_target(self.data[:, -1]).astype(np.float32)
        self.users_dims = np.max(self.users,axis=0)+1
        self.items_dims = np.max(self.items, axis=0) + 1
        self.user_offsets = np.array((0, *np.cumsum(self.users_dims[:-1])), dtype=np.int64)
        self.item_offsets = np.array((0, *np.cumsum(self.items_dims[:-1])), dtype=np.int64)
        self.users_input = self.users+self.user_offsets
        self.items_input = self.items+self.item_offsets


    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, index):
        return self.users_input[index], self.items_input[index],  self.target[index]

    def __preprocess_target(self, target):

        target[target <= 3] = 0
        target[target > 3] = 1
        return target


# a = train_data("./data_pre.txt")
# print(a.data[0])
# print(a.users[0])
# print(a.items[0])
# print(a.target[0])
# print(len(a))
# print(a.users_dims)
# print(a.items_dims)
# print(a.users_input[0])
# print(a.items_input[0])









