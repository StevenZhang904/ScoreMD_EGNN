import os
import csv
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

class RMSD(object):
    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir
        self.device = self._get_device()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def compute_loss(self, y_true, y_pred):
        atom_num = y_true.shape[0]
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        msd = torch.sum((y_true - y_pred)**2) / atom_num
        rmsd = torch.sqrt(msd)
        return rmsd
    
    def plot(self, loss_ls):
        x = np.arange(len(loss_ls))

        fig = plt.figure(figsize=[4,3], dpi=600)

        # set the basic properties
        plt.xlabel('# snapshots', fontsize=12)
        plt.ylabel('RMSD', fontsize=12)

        plt.plot(x, loss_ls, color='b')
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.tick_params(direction='in', width=1.5)

        # plt.show()
        plt.savefig(os.path.join(self.save_dir, 'rmsd.png'), dpi=600, bbox_inches='tight')

    