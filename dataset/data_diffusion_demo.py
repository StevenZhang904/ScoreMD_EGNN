import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
from copy import deepcopy
import mdtraj as md
import torch
from torch.utils.data import Dataset, DataLoader

class Diffusion_Dataset(Dataset):
    def __init__(self, data_dir: str, sys_dir = None, name = "circular_22_"):
        super(Diffusion_Dataset, self).__init__()
        self.data_dir = data_dir
        self.name = name
        sys_data = pd.read_csv("/home/cmu/Desktop/Summer_research/ScoreMD_EGNN/sys_data.csv")
        sys_file_names = sys_data["sys_filename"]
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

        mean, std, var = np.mean(positions), np.std(positions), np.var(positions)
        print(mean, std, var)
        positions = (positions - mean)/std
        mean, std, var = np.mean(positions), np.std(positions), np.var(positions)
        print(mean, std, var)            
                
        ## TODO: normalize displacements if needed
        
        self.positions = np.array(positions, dtype = np.float32)
        self.labels = np.array(displacements, dtype = np.float32)
        self.len = len(self.labels)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.positions[index] , self.labels[index]
    



Diffusion_Data = Diffusion_Dataset(
    data_dir = "/home/cmu/Desktop/Summer_research/position_data_2.csv",
    sys_dir = "/home/cmu/Desktop/Summer_research/sys_pdb/",
    name = "circular_22_",
)

train_size = int(0.8 * len(Diffusion_Data))
test_size = len(Diffusion_Data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(Diffusion_Data, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
e = 0
print(len(train_dataloader))
print(len(next(enumerate(train_dataloader))[1][1]))
