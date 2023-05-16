import os
import yaml
from copy import deepcopy
import numpy as np
import torch

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

from models.egnn import EGNN
from dataset.data_sample import MDSampleDatasetWrapper
from utils.ema import EMAHelper
from utils.md import get_neighbor
from utils.denoising import get_beta_schedule, Normalizer
from utils.rdf import Tester
from utils.rmsd import RMSD
from datetime import datetime
import shutil


class Sampler(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self._load_model()

        self.dataset = MDSampleDatasetWrapper(**config['sample_dataset'])

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
        
        # self.save_dir = config['load_model'].split("/")[-1]
        self.save_dir = os.path.join(config['save_dir'], datetime.now().strftime('%b%d_%H-%M-%S'))
        os.makedirs(self.save_dir, exist_ok=True)
        print("Save dir:", self.save_dir)

        self._save_config_file(self.save_dir)

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
        shutil.copy('./config_sample.yaml', os.path.join(ckpt_dir, 'config_sample.yaml'))
    
    def _load_model(self):
        self.model = EGNN(**self.config["model"])
        self.model = self.model.to(self.device)
        # self.normalizer_disp = Normalizer(torch.zeros(3))
        # self.normalizer_vel = Normalizer(torch.zeros(3))
        self.normalizer_acc = Normalizer(torch.zeros(3))

        # load model & other parameters
        ckpt_path = os.path.join(self.config['load_model'], "model.ckpt")
        states = torch.load(ckpt_path, map_location=self.device)
        # print(states[0])
        self.model.load_state_dict(states[0], strict=True)
        # self.model.load_pl_state_dict(states[0])
        print("Model loaded from:", ckpt_path)

        # self.normalizer_disp.load_state_dict(states[1])
        # self.normalizer_vel.load_state_dict(states[2])
        self.normalizer_acc.load_state_dict(states[1])
        print("Normalizer loaded from:", ckpt_path)

        if self.config['diffusion']['ema']:
            ema_helper = EMAHelper(mu=self.config['diffusion']['ema_rate'])
            ema_helper.register(self.model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(self.model)
        else:
            ema_helper = None

    def sample(self):
        self.dataloader = self.dataset.get_data_loaders()

        for bn, data in enumerate(self.dataloader):
            data_init = data.to(self.device)
            break

        self.data_curr = deepcopy(data_init)

        traj_idx = data_init.traj_idx
        frame_idx = data_init.frame_idx
        sample_frames = 1 + self.config["diffusion"]["sample_num"]
        if frame_idx + sample_frames + self.dataloader.dataset.interval < self.dataloader.dataset.traj_len:
            gt_frames = sample_frames
        else:
            gt_frames = self.dataloader.dataset.traj_len - frame_idx - self.dataloader.dataset.interval - 1
        gt_pos = self.dataloader.dataset.wrap_trajs[traj_idx][frame_idx:frame_idx + gt_frames] * 10
        gt_pos = torch.tensor(gt_pos)

        gt_unwrap = self.dataloader.dataset.unwrap_trajs[traj_idx][frame_idx:frame_idx + gt_frames] * 10
        gt_unwrap = torch.tensor(gt_unwrap)

        self.model.eval()
        self.sample_trajectory(sample_frames, gt_pos, gt_unwrap)

    def sample_trajectory(self, sample_frames, gt_pos):
        # x = torch.randn(
        #     size=self.data_curr.pos.shape,
        #     device=self.device,
        # )

        feat = deepcopy(self.data_curr.feat)
        self.sample_traj = deepcopy(self.dataloader.dataset.wrap_trajs[0])

        print("Sample frames:", self.sample_traj.shape)
        xyz = torch.zeros((sample_frames, self.sample_traj.shape[1], 3))    # store the sample position
        xyz[0] = self.data_curr.pos
        vel = self.data_curr.vel

        # self.data_curr.disp = torch.randn_like(self.data_curr.pos)

        rmsd_ls = []      # restore the rmsd for each sampled snapshot
        unwrap_pos = self.data_curr.unwrap_pos
        rmsd = RMSD(self.config, self.save_dir)

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            for i in range(sample_frames - 1):
                print("Generating step:", i)
                x = torch.randn(
                    size=self.data_curr.pos.shape,
                    device=self.device,
                )

                acc = self.sample_snapshot(x, self.model, last=True)               # Predict acc
                acc = self.normalizer_acc.denorm(acc)                              # Normalize acc
                vel_new = acc + vel                                                # new velocity
                dt = 1000 * self.config['dataset']['interval']                     # timestep in fs
                dx = (vel + vel_new) * dt / 2                                      # displacement
                pos = self.data_curr.pos + dx                                      # Unwrap position
                unwrap_pos += dx
                pos_wrap = torch.remainder(pos, self.dataloader.dataset.box_size)  # wrap to the box according to PBC
                print('pos before MD:', pos_wrap[:9, :])

                # if i % 5 == 0:
                print("Calling MD for refinement")
                pos_wrap = self.md_step(steps=10, sampled_pos=pos_wrap, sampled_vel=vel_new)
                print('pos after MD:', pos_wrap[:9, :])
                
                vel = vel_new
                self.data_curr.pos = pos_wrap

                xyz[i + 1] = pos_wrap.detach().cpu()

                if i + 1 < gt_unwrap.shape[0]:
                    rmsd_ls.append(rmsd.compute_loss(gt_unwrap[i + 1], unwrap_pos).detach().cpu().numpy())
                else:
                    continue

        torch.save(xyz / 10, self.save_dir + "/sample_traj.pt")                     # Angstrom to nanometer
        print("Sampled trajectory saved to:", self.save_dir + "/sample_traj.pt")

        # save the sampled trajectory via rewriting MDTraj
        save_traj = self.dataloader.dataset.template_traj
        save_traj.xyz[:sample_frames] = xyz[:sample_frames].numpy() / 10            # Angstrom to nanometer
        sample_traj_path = self.save_dir + "/sample_traj.lammpstrj"
        save_traj.save_lammpstrj(sample_traj_path)
        print("Sampled trajectory saved to:", self.save_dir + "/sample_traj.lammpstrj")

        box_size = self.dataloader.dataset.box_size

        tester = Tester(self.config, xyz[:sample_frames], gt_pos, box_size, self.save_dir)
        tester.test()

        rmsd.plot(rmsd_ls)

    def sample_snapshot(self, x, model, last=True):
        try:
            # skip = self.args.skip
            skip = self.config["diffusion"]["skip"]
        except Exception:
            skip = 1

        if self.config["diffusion"]["sample_type"] == "generalized":
            if self.config["diffusion"]["skip_type"] == "uniform":
                skip = self.num_timesteps // self.config["diffusion"]["timesteps"]
                seq = range(0, self.num_timesteps, skip)
            elif self.config["diffusion"]["skip_type"] == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.config["diffusion"]["timesteps"]
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            
            from utils.denoising import generalized_steps
            xs = generalized_steps(
                x, seq, model, self.betas, self.data_curr, self.device, 
                self.normalizer_acc,
                eta=self.config["diffusion"]["eta"])
            x = xs

        elif self.config["diffusion"]["sample_type"] == "ddpm_noisy":
            if self.config["diffusion"]["skip_type"] == "uniform":
                skip = self.num_timesteps // self.config["diffusion"]["timesteps"]
                seq = range(0, self.num_timesteps, skip)
            elif self.config["diffusion"]["skip_type"] == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.config["diffusion"]["timesteps"]
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError

            from utils.denoising import ddpm_steps
            x = ddpm_steps(
                x, seq, model, self.betas, self.data_curr, self.device, 
                self.normalizer_acc
            )

        else:
            raise NotImplementedError

        if last:
            x = x[0][-1]

        return x

    def md_step(self, steps, sampled_pos, sampled_vel):
        pdb_path = os.path.join(self.config['dataset']['data_dir'], 'wb_2.pdb')
        save_path = os.path.join(self.save_dir, 'md.pdb')
        pdb = PDBFile(pdb_path)

        coords = []
        for i in range(len(pdb.positions)):
            coords.append(Vec3(sampled_pos[i, 0], sampled_pos[i, 1], sampled_pos[i, 2]))
        pdb.positions = coords * angstroms
        pdb._positions = [pdb.positions]
        pdb.topology.createDisulfideBonds(pdb.positions)

        velocities = []
        for i in range(len(pdb.positions)):
            velocities.append(Vec3(sampled_vel[i, 0], sampled_vel[i, 1], sampled_vel[i, 2]))
        velocities = velocities * angstroms/femtoseconds
        print(sampled_vel[:9])

        forcefield = ForceField('./spc.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
                nonbondedCutoff=1*nanometer, constraints=HBonds)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 1*femtoseconds)
        # integrator = NoseHooverIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocities(velocities)
        print(simulation.context.getState(getVelocities=True).getVelocities()[:9])
        simulation.minimizeEnergy()
        simulation.reporters.append(PDBReporter(save_path, steps))
        simulation.reporters.append(StateDataReporter(stdout, steps, step=True,
                potentialEnergy=True, temperature=True))
        simulation.step(steps)

        new_pdb = PDBFile(save_path)
        new_pos = np.array(new_pdb.getPositions(asNumpy=True)) * 10
        new_pos = torch.tensor(new_pos, dtype=torch.float, device=self.device)
        return new_pos


def main():
    config = yaml.load(
        open("config_sample.yaml", "r"), Loader=yaml.FullLoader
    )
    sample_config = deepcopy(config)
    train_config = yaml.load(
        open(os.path.join(config['load_model'], 'config.yaml'), "r"), 
        Loader=yaml.FullLoader
    )
    config.update(train_config)
    config['gpu'] = sample_config['gpu']
    config['load_model'] = sample_config['load_model']
    config['diffusion']['sample_num'] = sample_config['diffusion']['sample_num']
    config['sample_dataset']['data_dir'] = train_config['dataset']['data_dir']
    config['sample_dataset']['interval'] = train_config['dataset']['interval']
    config['sample_dataset']['seed'] = train_config['dataset']['seed']
    # config['sample_dataset']['valid_size'] = train_config['dataset']['valid_size']
    print(config)

    sampler = Sampler(config)
    sampler.sample()


if __name__ == "__main__":
    main()