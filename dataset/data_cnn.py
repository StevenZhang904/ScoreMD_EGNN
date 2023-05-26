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
from torchvision import transforms
from torchvision.io import read_image


class CNN_Dataset(Dataset):
    def __init__(self, sys_data_dir: str, mem_data_dir: str, transform = None):
        super(CNN_Dataset, self).__init__()
        self.sys_data_dir = sys_data_dir
        self.mem_data_dir = mem_data_dir ### In images (png) format
        self.transform = transform

        self.imgs_dir, labels = [], []
        sys_data = pd.read_csv(self.sys_data_dir)
        water_counts = torch.tensor(sys_data["water_counts"].values) 
        sys_file_names = sys_data["sys_filename"].values
        mem_img_list = os.listdir(mem_data_dir)

        for x in range(len(sys_file_names)):
            sys_file_name = sys_file_names[x]
            for y in mem_img_list:
                if sys_file_name == y[:-4]:
                    img_dir = mem_data_dir+y
                    self.imgs_dir.append(img_dir)
                    labels.append(water_counts[x])

        self.labels = np.array(labels, dtype = np.float32)
        self.len = len(self.labels)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        img = Image.open(self.imgs_dir[index])
        ## Raw images has four channels, converted to RGB to become 3 channels
        img = img.convert("RGB")
        img = self.transform(img)
        img = np.array(img) / 255.0
        img = np.moveaxis(img, 2, 0)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        label = self.labels[index]
        return img , label
    



class Diffusion_Demo_Dataset(Dataset):
    def __init__(self, sys_data_dir: str, mem_data_dir: str, transform = None):
        super(Diffusion_Demo_Dataset, self).__init__()
        self.sys_data_dir = sys_data_dir
        self.mem_data_dir = mem_data_dir ### In images (png) format
        self.transform = transform

        self.imgs_dir, labels = [], []
        sys_data = pd.read_csv(self.sys_data_dir)
        displacements = torch.tensor(sys_data[1].values)
        # water_counts = torch.tensor(sys_data["water_counts"].values) 
        sys_file_names = sys_data["sys_filename"].values
        # mem_img_list = os.listdir(mem_data_dir)

        for x in range(len(sys_file_names)):
            sys_file_name = sys_file_names[x]
            y = "circular_22"
            if sys_file_name == y:
                # img_dir = mem_data_dir+ y
                # self.imgs_dir.append(img_dir)
                labels.append(displacements[x])

        self.labels = np.array(labels, dtype = np.float32)
        self.len = len(self.labels)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        img = Image.open(self.imgs_dir[index])
        ## Raw images has four channels, converted to RGB to become 3 channels
        img = img.convert("RGB")
        img = self.transform(img)
        img = np.array(img) / 255.0
        img = np.moveaxis(img, 2, 0)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        label = self.labels[index]
        return img , label




            