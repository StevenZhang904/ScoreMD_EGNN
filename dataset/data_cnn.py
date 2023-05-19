import os
import h5py
import random
from typing import Dict, List
import numpy as np
import pandas as pd
from copy import deepcopy
import mdtraj as md
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from torchvision.io import read_image


class DiffusionDataset(Dataset):
    def __init__(self, sys_data_dir: str, mem_data_dir: str):
        super(DiffusionDataset, self).__init__()
        self.sys_data_dir = sys_data_dir
        self.mem_data_dir = mem_data_dir ### In images (png) format

        imgs, labels = [], []
        sys_data = pd.read_csv("./sys_data.csv")
        water_counts = torch.tensor(sys_data["num_water"].values)
        sys_file_names = torch.tensor(sys_data["sys_file_name"].values)

        mem_img_list = os.listdir(mem_data_dir)

        for x in range(len(sys_file_names)):
            sys_file_name = sys_file_names[x]
            for y in mem_img_list:
                if sys_file_name == y[:-4]:
                    img = read_image(mem_data_dir+y)
                    imgs.append(img)
                    labels.append(water_counts[x])

        
        


    
    def __getitem__(self, index):
        img = 
        






            