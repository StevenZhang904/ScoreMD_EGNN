import os
import random
from typing import Dict, List
import numpy as np
import mdtraj as md

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader


class MDSampleDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        traj_indices: List[int],
        interval: int,
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

            self.wrap_trajs.append(np.array(wrap_traj.xyz) * 10)     # nm to Angstrom
            self.unwrap_trajs.append(np.array(unwrap_traj.xyz) * 10) # nm to Angstrom
            self.vels.append(np.array(vel_traj.xyz) * 10)            # nm/fs to Angstrom/fs
        
        self.template_traj = wrap_traj
        self.n_traj = len(traj_indices)
        self.traj_len = wrap_traj.xyz.shape[0]
        self.box_size = np.mean(wrap_traj.unitcell_lengths[0]) * 10 # nm to Angstrom
        print('loaded # trajectories:', self.n_traj)
        print('trajectory length:', self.traj_len)
        print('Box size:', self.box_size, 'Angstrom')

    def __getitem__(self, index):
        """
        Units (https://docs.lammps.org/units.html):
            pos: nanometer (10^(-9) meter)
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
        disp = disp

        feat = []
        for i in range(pos.shape[0]):
            if i % 3 == 0:
                feat.append(1)
            else:
                feat.append(0)
        
        row, col = [], []
        for i in range(pos.shape[0] // 3):
            row += [3*i, 3*i+1]
            col += [3*i+1, 3*i]
            row += [3*i, 3*i+2]
            col += [3*i+2, 3*i]

        feat = torch.tensor(feat, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        unwrap_pos = torch.tensor(unwrap_pos, dtype=torch.float)
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
            feat=feat, pos=pos, disp=disp, vel=vel, acc=acc, unwrap_pos=unwrap_pos,
            edge_index=edge_index, traj_idx=traj_idx, frame_idx=frame_idx,
        )
        return data

    def __len__(self):
        return (self.traj_len - self.interval) * self.n_traj


class MDSampleDatasetWrapper(object):
    def __init__(
        self, batch_size, num_workers, 
        data_dir, interval, seed
    ):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.interval = interval
        self.seed = seed

    def get_data_loaders(self):
        traj_indices = set()
        for f in os.listdir(self.data_dir):
            if f.endswith('.lammpstrj'):
                idx = int(f.split('_')[-1].replace('.lammpstrj', ''))
                traj_indices.add(idx)
        traj_indices = list(traj_indices)

        # n_traj = len(traj_indices)
        # random_state = np.random.RandomState(seed=self.seed)
        # random_state.shuffle(traj_indices)
        # split = max([int(np.floor(self.valid_size * n_traj)), 1])
        # valid_idx = traj_indices[:split]
        
        print('Building sampling dataset...')
        dataset = MDSampleDataset(self.data_dir, traj_indices, self.interval)
        
        loader = PyGDataLoader(
            dataset, batch_size=1, shuffle=True,
            num_workers=self.num_workers, drop_last=True, 
            pin_memory=True, persistent_workers=(self.num_workers > 0)
        )
        return loader