import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
from copy import deepcopy
import mdtraj as md
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

class Diffusion_Dataset(Dataset):
    def __init__(self, data_dir: str, sys_dir = None, name = "circular_22_"):
        super(Diffusion_Dataset, self).__init__()
        self.data_dir = data_dir
        self.name = name
        # sys_data = pd.read_csv("/home/cmu/Desktop/Summer_research/ScoreMD_EGNN/sys_data.csv")
        # sys_file_names = sys_data["sys_filename"]
        # data_dir = "/home/cmu/Desktop/Summer_research/position_data_2.csv"
        # sys_dir = "/home/cmu/Desktop/Summer_research/sys_pdb/"
        # name = "circular_22_"

        positions, displacements = [], []
        pos_2_data = pd.read_csv(self.data_dir)
        timestamps = pos_2_data["names"].values

        for x in range(len(timestamps)):
            file_name = name+str(timestamps[x])
            traj = md.load_pdb(sys_dir+file_name+".pdb")
            pos = traj.xyz[0].flatten()
            positions.append(pos)
            if x == 0:
                displacements.append(np.zeros_like(pos))
            else:
                displacement = pos - positions[x-1]
                displacements.append(displacement)    

        positions = np.array(positions, dtype = np.float32)
        mean, std, var = np.mean(positions, axis = 0), np.std(positions, axis = 0), np.var(positions, axis = 0)
        print(mean, std, var)
        positions = (positions - mean)/std
        # mean, std, var = np.mean(positions, axis = 0), np.std(positions, axis = 0), np.var(positions, axis = 0)
        # print(mean, std, var)            
                
        ## TODO: normalize displacements if needed
        print(type(positions[1]), type(positions))
        self.positions = np.array(positions, dtype = np.float32)
        self.labels = np.array(displacements, dtype = np.float32)
        self.len = len(self.labels)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.positions[index] , self.labels[index]
    
class Diff_EGNN_Dataset(Dataset):
    def __init__(self, data_dir, sys_dir, name = "circular_22_"):
        self.data_dir = data_dir
        data = pd.read_csv(self.data_dir)
        timestamps = data["names"].values
        positions, displacements = [], []

        for x in range(len(timestamps)):
            file_name = name+str(timestamps[x])
            traj = md.load_pdb(sys_dir+file_name+".pdb")
            pos = traj.xyz[0].flatten()
            positions.append(pos)
            if x == 0:
                displacements.append(np.zeros_like(pos))
            else:
                displacement = pos - positions[x-1]
                displacements.append(displacement)    

        positions = np.array(positions, dtype = np.float32)

        mean, std = positions.mean(), positions.std()
        print(mean, std)
        positions = (positions - mean)/std  ### Normalize to (0, 1) range
        positions = positions*2 - 1 ### Normalize to (-1,1) range as noted in the DDPM paper

        self.positions = np.array(positions, dtype = np.float32)
        self.len = len(self.positions)
        ### Hard code atom types: where 0 is oxygen and 1 is hydrogen
        x = [0, 1, 1, 0, 1, 1]
        x = torch.tensor(x, dtype=torch.long)
        self.x = x
        
    def __getitem__(self, index):
        pos = self.positions[index]
        pos = pos.reshape(6,3)
        pos = torch.tensor(pos, dtype=torch.float)
        data = Data(x=self.x, pos=pos)
        return data

    def __len__(self):
        return self.len

class Diff_EGNN_wrapper(object):
    def __init__(self, batch_size, num_workers, data_dir, sys_dir, name, seed = None):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.sys_dir = sys_dir
        self.name = name

    def get_data_loaders(self):
        data = Diff_EGNN_Dataset(data_dir=self.data_dir, sys_dir = self.sys_dir, name = self.name)
        train_size = int(0.9 * len(data))
        test_size = len(data) - train_size  

        train_valid_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
        # valid_size = train_size*0.1
        # train_size = train_size - valid_size

        # train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [train_size, valid_size])

        train_loader = PyGDataLoader( 
            train_valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=True, 
            pin_memory=True, persistent_workers=False
        )
        # valid_loader = PyGDataLoader(
        #     valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
        #     shuffle=True, drop_last=True, 
        #     pin_memory=True, persistent_workers=False
        # )
        test_loader = PyGDataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=True, 
            pin_memory=True, persistent_workers=False
        )
        return train_loader, test_loader

# Diffusion_Data = Diffusion_Dataset(
#     data_dir = "/home/cmu/Desktop/Summer_research/position_data_2.csv",
#     sys_dir = "/home/cmu/Desktop/Summer_research/sys_pdb/",
#     name = "circular_22_",
# )
# train_size = int(0.8 * len(Diffusion_Data))
# test_size = len(Diffusion_Data) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(Diffusion_Data, [train_size, test_size])
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
# e = 0
# print(len(train_dataloader))
# print(len(next(enumerate(train_dataloader))[1][1]))



batch_size = 4
Diffusion_Data = Diff_EGNN_wrapper(
    batch_size = batch_size,
    num_workers = 7,
    data_dir = "/home/cmu/Desktop/Summer_research/position_data_2.csv",
    sys_dir = "/home/cmu/Desktop/Summer_research/sys_pdb/",
    name = "circular_22_", 
)
train_dataloader, test_dataloader = Diffusion_Data.get_data_loaders()
print(len(train_dataloader))
print(next(enumerate(train_dataloader)))
print(next(enumerate(train_dataloader))[1].ptr)
print(len(next(enumerate(train_dataloader))[1].batch))
