import os
import csv
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md

# from dataset.data_tip3p_unwrap import MDDatasetWrapper
from dataset.data_spc import MDDatasetWrapper
from models.egnn_acc import EGNN
from utils.md import RDF, RDF2Sys


# class Tester(object):
#     def __init__(self, config):
#         self.config = config
#         self.device = self._get_device()
#         self.dataset = MDDatasetWrapper(**config['dataset'])

#     def _get_device(self):
#         if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
#             device = self.config['gpu']
#         else:
#             device = 'cpu'
#         print("Running on:", device)

#         return device
    
#     def test(self):
#         __, test_loader = self.dataset.get_data_loaders()

#         model = EGNN(**self.config["model"])
#         model = model.to(self.device)
#         model_path = os.path.join(self.config['load_model'], 'checkpoints', 'model.pth')
#         state_dict = torch.load(model_path, map_location=self.device)
#         model.load_state_dict(state_dict)
#         print("Loaded {} with success.".format(model_path))

#         box_size = test_loader.dataset.box_size
#         rdf_model = RDF(nbins=150, r_range=(0., box_size/2.)).to(self.device)
#         rdf2_model = RDF2Sys(nbins=150, r_range=(0., box_size/2.)).to(self.device)

#         rdf_gt_np, rdf_np= 0., 0.
#         rdf2_gt_np, rdf2_np = 0., 0.

#         with torch.no_grad():
#             model.eval()

#             for bn, data in enumerate(test_loader):                
#                 data = data.to(self.device)
                
#                 pred_disp = model(
#                     data.feat, data.pos, data.batch, 
#                     data.edge_index, data.edge_vec, data.edge_weight
#                 )

#                 pos_gt = data.pos + data.disp
#                 pos_pred = data.pos + pred_disp

#                 __, __, rdf_gt = rdf_model(pos_gt[::3])
#                 count, bins, rdf = rdf_model(pos_pred[::3])

#                 __, __, rdf2_gt = rdf2_model(pos_gt[::3], pos_gt[1::3])
#                 count2, bins2, rdf2 = rdf2_model(pos_pred[::3], pos_pred[1::3])

#                 rdf_gt_np += rdf_gt.numpy()
#                 rdf_np += rdf.numpy()

#                 rdf2_gt_np += rdf2_gt.numpy()
#                 rdf2_np += rdf2.numpy()

#         fig = plt.figure(figsize=[4,3], dpi=600)

#         # set the basic properties
#         plt.xlabel('$r_{OO} / \AA$', fontsize=12)
#         plt.ylabel('$g(r_{OO})$', fontsize=12)

#         plt.plot(bins[:-3], rdf_gt_np[:-2], label='GT', color='b')
#         plt.plot(bins[:-3], rdf_np[:-2], label='Pred', color='r')
#         ax = plt.gca()
#         for axis in ['top','bottom','left','right']:
#             ax.spines[axis].set_linewidth(2)
#         plt.tick_params(direction='in', width=1.5)
#         plt.legend(loc='lower right', fontsize=12, frameon=False)

#         # plt.show()
#         plt.savefig(os.path.join(self.config['load_model'], 'rdf.png'), dpi=600, bbox_inches='tight')

#         fig = plt.figure(figsize=[4,3], dpi=600)

#         # set the basic properties
#         plt.xlabel('$r_{OH} / \AA$', fontsize=12)
#         plt.ylabel('$g(r_{OH})$', fontsize=12)

#         plt.plot(bins2[:-6], rdf2_gt_np[:-5], label='GT', color='b')
#         plt.plot(bins2[:-6], rdf2_np[:-5], label='Pred', color='r')
#         ax = plt.gca()
#         for axis in ['top','bottom','left','right']:
#             ax.spines[axis].set_linewidth(2)
#         plt.tick_params(direction='in', width=1.5)
#         plt.legend(loc='lower right', fontsize=12, frameon=False)

#         # plt.show()
#         plt.savefig(os.path.join(self.config['load_model'], 'rdf2.png'), dpi=600, bbox_inches='tight')

#         # with open(os.path.join(self.log_dir, 'results.csv'), mode='w', newline='') as csv_file:
#         #     csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         #     for i in range(len(labels)):
#         #         csv_writer.writerow([''.join(smiles[i]), predictions[i], labels[i]])
#         #     csv_writer.writerow([mean_squared_error(labels, predictions, squared=False)])

class Tester(object):
    def __init__(self, config, sample_pos, gt_pos, box_size, save_dir):
        self.config = config    
        self.device = self._get_device() 
        self.sample_pos = sample_pos.to(self.device)
        self.gt_pos = gt_pos.to(self.device)
        self.box_size = box_size 
        self.save_dir = save_dir      

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device
    
    def test(self):
        rdf_model = RDF(nbins=150, r_range=(0., self.box_size/2.)).to(self.device)
        rdf2_model = RDF2Sys(nbins=150, r_range=(0., self.box_size/2.)).to(self.device)

        rdf_gt_np, rdf_np= 0., 0.
        rdf2_gt_np, rdf2_np = 0., 0.

        for i in range(self.sample_pos.shape[0]):
            pos_sample = self.sample_pos[i]

            count, bins, rdf = rdf_model(pos_sample[::3])
            count2, bins2, rdf2 = rdf2_model(pos_sample[::3], pos_sample[1::3])
        
            rdf_np += rdf.numpy() / self.sample_pos.shape[0]
            rdf2_np += rdf2.numpy() / self.sample_pos.shape[0]

        for i in range(self.gt_pos.shape[0]):
            pos_gt = self.gt_pos[i]

            __, __, rdf_gt = rdf_model(pos_gt[::3])
            __, __, rdf2_gt = rdf2_model(pos_gt[::3], pos_gt[1::3])

            rdf_gt_np += rdf_gt.numpy() / self.gt_pos.shape[0]
            rdf2_gt_np += rdf2_gt.numpy() / self.gt_pos.shape[0]

        fig = plt.figure(figsize=[4,3], dpi=600)

        # set the basic properties
        plt.xlabel('$r_{OO} / \AA$', fontsize=12)
        plt.ylabel('$g(r_{OO})$', fontsize=12)

        plt.plot(bins[:-3], rdf_gt_np[:-2], label='GT', color='b')
        plt.plot(bins[:-3], rdf_np[:-2], label='Pred', color='r')
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.tick_params(direction='in', width=1.5)
        plt.legend(loc='lower right', fontsize=12, frameon=False)

        # plt.show()
        plt.savefig(os.path.join(self.save_dir, 'rdf.png'), dpi=600, bbox_inches='tight')

        fig = plt.figure(figsize=[4,3], dpi=600)

        # set the basic properties
        plt.xlabel('$r_{OH} / \AA$', fontsize=12)
        plt.ylabel('$g(r_{OH})$', fontsize=12)

        plt.plot(bins2[:-6], rdf2_gt_np[:-5], label='GT', color='b')
        plt.plot(bins2[:-6], rdf2_np[:-5], label='Pred', color='r')
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.tick_params(direction='in', width=1.5)
        plt.legend(loc='lower right', fontsize=12, frameon=False)

        # plt.show()
        plt.savefig(os.path.join(self.save_dir, 'rdf2.png'), dpi=600, bbox_inches='tight')

        # with open(os.path.join(self.log_dir, 'results.csv'), mode='w', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     for i in range(len(labels)):
        #         csv_writer.writerow([''.join(smiles[i]), predictions[i], labels[i]])
        #     csv_writer.writerow([mean_squared_error(labels, predictions, squared=False)])

if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    tester = Tester(config)
    tester.test()