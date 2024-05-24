import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd


class train_data(torch.utils.data.Dataset):
    def __init__(self,dataset_path):
        # data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        data_file = pd.read_csv(dataset_path, sep = ',', engine='python', header='infer')
        new_columns = ["gender", "age", "occupation", "user_id", "genre", "movie_id", "score"]
        data_file = data_file.reindex(columns=new_columns)
        self.data = data_file.to_numpy()
        self.items = self.data[:, :-1].astype(int)
        self.targets = self.__preprocess_target(self.data[:, -1]).astype(np.float32)
        self.field_dims=np.max(self.items,axis=0)+1


    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):

        target[target <= 3] = 0
        target[target > 3] = 1
        return target














