import os
import gc
import tqdm
import yaml
import shutil
import numpy as np
from datetime import datetime

import torch
from torch.optim import AdamW
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter

from models.egnn import EGNN
from dataset.data_spc import MDDatasetWrapper, SampleDatasetWrapper
from utils.ema import EMAHelper
from utils.lr_sched import adjust_learning_rate
from utils.md import get_neighbor
from utils.denoising import get_beta_schedule, Normalizer, loss_registry


class Diffusion(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        
        self.log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self._save_config_file(self.log_dir)

        self.dataset = MDDatasetWrapper(**config['dataset'])
        # self.sample_dataset = SampleDatasetWrapper(**config['sample_dataset'], cutoff=config['model']['cutoff_upper'])

        self.model_var_type = config['diffusion']['var_type']
        betas = get_beta_schedule(
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            num_diffusion_timesteps=config['diffusion']['num_diffusion_timesteps'],
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    @staticmethod
    def _save_config_file(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        shutil.copy('./config.yaml', os.path.join(ckpt_dir, 'config.yaml'))

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        displacements, velocities = [], []
        for i, d in enumerate(train_loader):
            displacements.append(d.disp)
            velocities.append(d.vel)
            if (i + 1) % 500 == 0:
                print('normalizing', i)
                # break
        displacements = torch.cat(displacements).view(-1)
        velocities = torch.cat(velocities).view(-1)
        self.normalizer_disp = Normalizer(displacements)
        self.normalizer_vel = Normalizer(velocities)
        del displacements, velocities
        gc.collect() # free memory
        print('displacement normalizer mean {}, std {}'.format(self.normalizer_disp.mean, self.normalizer_disp.std))
        print('velocity normalizer mean {}, std {}'.format(self.normalizer_vel.mean, self.normalizer_vel.std))

        model = EGNN(**self.config["model"])
        try:
            state_dict = torch.load(self.config['load_model'], map_location=self.device)
            model.load_state_dict(state_dict)
            print("Resume training from {}.".format(self.config['load_model']))
        except FileNotFoundError:
            print("No existing weights are found. Training from scratch.")
        model = model.to(self.device)

        if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
        # if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
        if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay']) 
        optimizer = AdamW(
            model.parameters(), self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )
        
        if self.config['diffusion']['ema']:
            ema_helper = EMAHelper(mu=self.config['diffusion']['ema_rate'])
            ema_helper.register(model)
        else:
            ema_helper = None
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            model.train()
            for bn, data in enumerate(train_loader):
                # adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)

                data = data.to(self.device)
                e = torch.randn_like(data.acc)
                b = self.betas
                n = data.num_snapshots.shape[0]

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                # reshape t to PyG format
                real_t = []
                curr_batch = 0
                for i in range(data.pos.shape[0]):
                    if data.batch[i] != curr_batch:
                        curr_batch = data.batch[i]
                    real_t.append(t[curr_batch])
                real_t = torch.tensor(real_t, dtype=torch.long).to(self.device)

                loss = loss_registry[self.config['diffusion']['type']](
                    model, self.normalizer_disp, self.normalizer_vel,
                    data, real_t, e, b, keepdim=False, reduce_mean=True,
                    num_diffusion_timesteps=self.config['diffusion']['num_diffusion_timesteps']
                )

                optimizer.zero_grad()
                loss.backward()
                # apply gradient clipping if defined
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config['grad_clip']
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config['diffusion']['ema']:
                    ema_helper.update(model)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=n_iter)
                    print(epoch_counter, bn, 'loss', loss.item())

                n_iter += 1

            # validate the model
            print("start validation")
            valid_loss = self._validate(model, valid_loader)
            self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
            print('Validation', epoch_counter, 'valid loss', valid_loss)

            states = [
                model.state_dict(),
                optimizer.state_dict(),
                self.normalizer_disp.state_dict(),
                self.normalizer_vel.state_dict(),
                train_loader.dataset.box_size,
                train_loader.dataset.interval,
                epoch_counter,
                n_iter
            ]
            if self.config['diffusion']['ema']:
                states.append(ema_helper.state_dict())

            torch.save(
                states,
                os.path.join(self.log_dir, "ckpt.pth".format(epoch_counter)),
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(states, os.path.join(self.log_dir, "ckpt_best.pth"))

            valid_n_iter += 1

    def _validate(self, model, valid_loader):
        valid_loss = 0.0
        with torch.no_grad():
            model.eval()

            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)
                e = torch.randn_like(data.acc)
                b = self.betas
                n = len(data.num_snapshots)

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                # reshape t to PyG format
                real_t = []
                curr_batch = 0
                for i in range(data.pos.shape[0]):
                    if data.batch[i] != curr_batch:
                        curr_batch = data.batch[i]
                    real_t.append(t[curr_batch])
                real_t = torch.tensor(real_t, dtype=torch.long).to(self.device)

                loss = loss_registry[self.config['diffusion']['type']](
                    model, self.normalizer_disp, self.normalizer_vel,
                    data, real_t, e, b, keepdim=False, reduce_mean=True
                )
                valid_loss += loss.item()
        
        # model.train()
        return valid_loss / (bn + 1)

    # def sample(self):
    #     model = TorchMD_ET(**self.config["model"])
    #     model = model.to(self.device)

    #     states = torch.load(self.config['load_model'], map_location=self.device)
    #     model.load_state_dict(states[0], strict=True)
    #     print("Load model successfully")

    #     if self.config['diffusion']['ema']:
    #         ema_helper = EMAHelper(mu=self.config['diffusion']['ema_rate'])
    #         ema_helper.register(model)
    #         ema_helper.load_state_dict(states[-1])
    #         ema_helper.ema(model)
    #     else:
    #         ema_helper = None

    #     dataloader = self.sample_dataset.get_data_loaders()

    #     for bn, data in enumerate(dataloader):
    #         data_init = data.to(self.device)
    #         break

    #     self.data_curr = data_init

    #     model.eval()
    #     self.sample_trajectory(model)

    # def sample_trajectory(self, model):
    #     config = self.config

    #     x = torch.randn(
    #         size=self.data_curr.pos.shape,
    #         device=self.device,
    #     )

    #     feat = self.data_curr.feat

    #     # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
    #     with torch.no_grad():
    #         # _, x = self.sample_snapshot(x, model, last=False)
    #         for i in range(self.config["diffusion"]["sample_num"]):
    #             print("Generate sample", i)
    #             disp = self.sample_snapshot(x, model, last=True)
    #             # print("shape of disp:", disp.shape)
    #             pos = self.data_curr.pos + disp
    #             edge_index, edge_vec, edge_weight, __ = get_neighbor(
    #                 pos, self.sample_dataset.Dataset.cutoff, self.sample_dataset.Dataset.box_size
    #             )
    #             self.data_curr = Data(
    #                 feat=feat, pos=pos, edge_index=edge_index,
    #                 edge_vec=edge_vec, edge_weight=edge_weight,
    #             )

    #     # x = [inverse_data_transform(config, y) for y in x]

    #     # for i in range(len(x)):
    #     #     for j in range(x[i].size(0)):
    #     #         tvu.save_image(
    #     #             x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
    #     #         )

    # def sample_snapshot(self, x, model, last=True):
    #     try:
    #         # skip = self.args.skip
    #         skip = self.config["diffusion"]["skip"]
    #     except Exception:
    #         skip = 1

    #     if self.config["diffusion"]["sample_type"] == "generalized":
    #         if self.config["diffusion"]["skip_type"] == "uniform":
    #             skip = self.num_timesteps // self.config["diffusion"]["timesteps"]
    #             seq = range(0, self.num_timesteps, skip)
    #         elif self.config["diffusion"]["skip_type"] == "quad":
    #             seq = (
    #                     np.linspace(
    #                         0, np.sqrt(self.num_timesteps * 0.8), self.config["diffusion"]["timesteps"]
    #                     )
    #                     ** 2
    #             )
    #             seq = [int(s) for s in list(seq)]
    #         else:
    #             raise NotImplementedError

    #         from utils.denoising import generalized_steps
    #         xs = generalized_steps(x, seq, model, self.betas, self.data_curr, self.device, eta=self.config["diffusion"]["eta"])
    #         x = xs

    #     elif self.config["diffusion"]["sample_type"] == "ddpm_noisy":
    #         if self.config["diffusion"]["skip_type"] == "uniform":
    #             skip = self.num_timesteps // self.config["diffusion"]["timesteps"]
    #             seq = range(0, self.num_timesteps, skip)
    #         elif self.config["diffusion"]["skip_type"] == "quad":
    #             seq = (
    #                     np.linspace(
    #                         0, np.sqrt(self.num_timesteps * 0.8), self.config["diffusion"]["timesteps"]
    #                     )
    #                     ** 2
    #             )
    #             seq = [int(s) for s in list(seq)]
    #         else:
    #             raise NotImplementedError

    #         from utils.denoising import ddpm_steps
    #         x = ddpm_steps(x, seq, model, self.betas, self.data_curr, self.device)

    #     else:
    #         raise NotImplementedError

    #     if last:
    #         x = x[0][-1]

    #     return x


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Diffusion(config)
    trainer.train()


if __name__ == "__main__":
    main()