import os
import h5py
import random
from typing import Dict, List
import numpy as np
from copy import deepcopy
import mdtraj as md

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

# from utils.md import get_neighbor


class MDDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        traj_indices: List[int],
        interval: int
    ):
        self.data_dir = data_dir
        self.interval = interval

        self.top_fn = None
        for f in os.listdir(self.data_dir):
            if f.endswith('.pdb'):
                self.top_fn = f
                break
        
        self.wrap_trajs, self.unwrap_trajs, self.vels = [], [], []

        for idx in traj_indices:
            wrap_fn = 'wb_{}.lammpstrj'.format(idx)
            unwrap_fn = 'wb_unwrap_{}.lammpstrj'.format(idx)
            vel_fn = 'wb_vel_{}.lammpstrj'.format(idx)

            wrap_traj = md.load(
                os.path.join(self.data_dir, wrap_fn), 
                top=os.path.join(self.data_dir, self.top_fn),
                stride=1
            )
            unwrap_traj = md.load(
                os.path.join(self.data_dir, unwrap_fn), 
                top=os.path.join(self.data_dir, self.top_fn),
                stride=1
            )
            vel_traj = md.load(
                os.path.join(self.data_dir, vel_fn), 
                top=os.path.join(self.data_dir, self.top_fn),
                stride=1
            )
            print('loaded trajectory:', wrap_fn)

            self.wrap_trajs.append(np.array(wrap_traj.xyz) * 10) # nm to Angstrom
            self.unwrap_trajs.append(np.array(unwrap_traj.xyz) * 10) # nm to Angstrom
            self.vels.append(np.array(vel_traj.xyz) * 10) # nm/fs to A/fs
        
        self.n_traj = len(traj_indices)
        self.traj_len = wrap_traj.xyz.shape[0]
        self.box_size = np.mean(wrap_traj.unitcell_lengths[0]) * 10 # nm to Angstrom
        print('loaded # trajectories:', self.n_traj)
        print('trajectory length:', self.traj_len)
        print('Box size:', self.box_size, 'Angstrom')

    def __getitem__(self, index):
        """
        Units (https://docs.lammps.org/units.html):
            pos: Angstroms (10^(-10) meter)
            velocity: Angstroms / femtosecond
            acceleration: Angstroms / (femtosecond*picosecond*interval)
            traj timestep: picosecond (1000 femtosecond)
            sample timestep: picosecond * interval
        """

        traj_idx = index // (self.traj_len - self.interval)
        frame_idx = index % (self.traj_len - self.interval)

        pos = self.wrap_trajs[traj_idx][frame_idx]
        vel = self.vels[traj_idx][frame_idx]
        next_vel = self.vels[traj_idx][frame_idx + self.interval]
        acc = next_vel - vel
        disp = self.unwrap_trajs[traj_idx][frame_idx + self.interval] - self.unwrap_trajs[traj_idx][frame_idx]

        feat = []
        for i in range(pos.shape[0]):
            if i % 3 == 0:
                feat.append(1)
                # feat.append(21)
            else:
                feat.append(0)
                # feat.append(10)
        
        row, col = [], []
        for i in range(pos.shape[0] // 3):
            row += [3*i, 3*i+1]
            col += [3*i+1, 3*i]
            row += [3*i, 3*i+2]
            col += [3*i+2, 3*i]

        feat = torch.tensor(feat, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        vel = torch.tensor(vel, dtype=torch.float)
        acc = torch.tensor(acc, dtype=torch.float)
        disp = torch.tensor(disp, dtype=torch.float)
        edge_index = torch.tensor([row, col], dtype=torch.long)

        cell = torch.tensor(
            [[self.box_size, 0, 0], [0, self.box_size, 0], [0, 0, self.box_size]],
            dtype=torch.float
        ).view(1, 3, 3)

        data = Data(
            num_snapshots=1, natoms=pos.size(0), cell=cell,
            feat=feat, pos=pos, disp=disp, vel=vel, acc=acc,
            edge_index=edge_index
        )
        return data

    def __len__(self):
        return (self.traj_len - self.interval) * self.n_traj


class MDDatasetWrapper(object):
    def __init__(
        self, batch_size, num_workers, valid_size, 
        data_dir, interval, seed
    ):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.interval = interval
        self.seed = seed

    def get_data_loaders(self):
        # n_traj = 0
        traj_indices = set()
        for f in os.listdir(self.data_dir):
            if f.endswith('.lammpstrj'):
                # n_traj = max([n_traj, int(f.split('_')[-1].replace('.lammpstrj', ''))])
                idx = int(f.split('_')[-1].replace('.lammpstrj', ''))
                # traj_indices.append(idx)
                traj_indices.add(idx)
            # if len(traj_indices) == 2:
            #     break
        traj_indices = list(traj_indices)

        # indices = list(range(1, n_traj + 1))
        n_traj = len(traj_indices)
        random_state = np.random.RandomState(seed=self.seed)
        random_state.shuffle(traj_indices)
        split = max([int(np.floor(self.valid_size * n_traj)), 1])
        valid_idx, train_idx = traj_indices[:split], traj_indices[split:]
        
        print('Building training dataset...')
        train_dataset = MDDataset(self.data_dir, train_idx, self.interval)
        print('Building validation dataset...')
        valid_dataset = MDDataset(self.data_dir, valid_idx, self.interval)
        
        train_loader = PyGDataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True, persistent_workers=(self.num_workers > 0)
        )
        valid_loader = PyGDataLoader(
            valid_dataset, batch_size=self.batch_size // 2, shuffle=False,
            num_workers=self.num_workers, drop_last=True, 
            pin_memory=True, persistent_workers=(self.num_workers > 0)
        )
        return train_loader, valid_loader


# class SampleDataset(Dataset):
#     def __init__(
#             self,
#             data_dir: str,
#             cutoff: float,
#             interval: int,
#             box_size: float,
#             from_start: bool = False,
#     ):
#         self.data_dir = data_dir
#         self.cutoff = cutoff
#         self.interval = interval
#         self.box_size = box_size
#         self.from_start = from_start

#         self.top_fn = None
#         for f in os.listdir(self.data_dir):
#             if f.endswith('.pdb'):
#                 self.top_fn = f
#                 break

#         for f in os.listdir(self.data_dir):
#             if f.startswith('wb_unwrap'):
#                 unwrap_fn = f
#             elif f.startswith('wb_vel'):
#                 vel_fn = f
#             else:
#                 wrap_fn = f
        
#         # self.wrap_trajs, self.unwrap_trajs, self.vels = [], [], []

#         wrap_traj = md.load(
#             os.path.join(self.data_dir, wrap_fn), 
#             top=os.path.join(self.data_dir, self.top_fn),
#             stride=1
#         )
#         # unwrap_traj = md.load(
#         #     os.path.join(self.data_dir, unwrap_fn), 
#         #     top=os.path.join(self.data_dir, self.top_fn),
#         #     stride=1
#         # )
#         vel_traj = md.load(
#             os.path.join(self.data_dir, vel_fn), 
#             top=os.path.join(self.data_dir, self.top_fn),
#             stride=1
#         )
#         print('loaded trajectory:', wrap_fn)

#         # self.wrap_trajs.append(np.array(wrap_traj.xyz))
#         # self.unwrap_trajs.append(np.array(unwrap_traj.xyz))
#         # self.vels.append(np.array(vel_traj.xyz))
#         self.template_traj = wrap_traj

#         self.wrap_traj = np.array(wrap_traj.xyz)
#         # self.unwrap_traj = np.array(unwrap_traj.xyz)
#         self.vel = np.array(vel_traj.xyz)
    
#         self.n_traj = 1
#         self.traj_len = wrap_traj.xyz.shape[0]
#         print('loaded # trajectories:', self.n_traj)
#         print('loaded trajectory length:', self.traj_len)

#     def __getitem__(self, index):
#         # traj_idx = index // (self.traj_len - self.interval)
#         # frame_idx = index % (self.traj_len - self.interval)

#         pos = self.wrap_traj[index]
#         vel = self.vel[index]

#         feat = []
#         for i in range(pos.shape[0]):
#             if i % 3 == 0:
#                 feat.append(1)
#             else:
#                 feat.append(0)
        
#         feat = torch.tensor(feat, dtype=torch.long)
#         pos = torch.tensor(pos, dtype=torch.float)
#         vel = torch.tensor(vel, dtype=torch.float)

#         cell = torch.tensor(
#             [[self.box_size, 0, 0], [0, self.box_size, 0], [0, 0, self.box_size]],
#             dtype=torch.float
#         )
#         data = Data(
#             num_snapshots=1, natoms=pos.size(0), feat=feat, pos=pos, vel=vel, cell=cell
#         )
#         return data

#         # edge_index, edge_vec, edge_weight, __ = get_neighbor(pos, self.cutoff, self.box_size)
#         # data = Data(
#         #     num_snapshots=1, natoms=pos.size(0), feat=feat, pos=pos, vel=vel,
#         #     edge_index=edge_index, edge_vec=edge_vec, edge_weight=edge_weight,
#         # )
#         # return data

#     def __len__(self):
#         if self.from_start:
#             return 1
#         return (self.traj_len - self.interval) * self.n_traj


# class SampleDatasetWrapper(object):
#     def __init__(
#         self, batch_size, num_workers, data_dir, 
#         cutoff, interval, box_size, from_start=False
#     ):
#         super(object, self).__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.cutoff = cutoff
#         self.interval = interval
#         self.box_size = box_size
#         self.from_start = from_start # whether to start from the beginning of the trajectory

#     def get_data_loaders(self):
#         print('Building sampling dataset...')

#         sample_dataset = SampleDataset(
#             self.data_dir, self.cutoff, self.interval, self.box_size, self.from_start
#         )
        
#         sample_loader = PyGDataLoader(
#             sample_dataset, batch_size=self.batch_size, shuffle=True,
#             num_workers=self.num_workers, drop_last=True,
#             pin_memory=True, persistent_workers=(self.num_workers > 0)
#         )
        
#         return sample_loader