import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import global_add_pool
import math

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
    return a


def generalized_steps(x, seq, model, b, data, device, normalizer_acc, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            xt = xs[-1].to(device)
            data.acc = normalizer_acc.norm(xt)
            __, et = model(data, t)
            # print('step:', i, j)
            # print(et[:5])

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to(device))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(device))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, data, device, normalizer_acc, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(device)

            data.acc = normalizer_acc.norm(x)
            __, output = model(data, t)
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to(device))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to(device))
    return xs, x0_preds


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

    elif beta_schedule == "cosine":
        # alpha_bar: a lambda that takes t from 0 to 1 and produces the cumulative product of (1-beta)
        # to that part of the diffusion process
        alpha_bar = lambda t: math.cos((t + 0.008) / (1 + 0.008) * math.pi /2) ** 2
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i+1) / num_diffusion_timesteps
            max_beta = 0.999 # smaller than 1 to prevent singularities
            betas.append(min(1-alpha_bar(t2) / alpha_bar(t1), max_beta))
        betas = np.array(betas)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def noise_estimation_loss(model,
                          normalizer_disp,
                          normalizer_vel,
                          data: torch_geometric.data.Data,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, 
                          keepdim=False,
                          reduce_mean=True,
                          num_diffusion_timesteps=1000
):
    x0 = data.disp
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1) # reshape a to the correct shape based on PyG Data
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt() # add noise to displacement

    data.disp = normalizer_disp.norm(x)
    data.vel = normalizer_vel.norm(data.vel)

    t = t.float()
    t *= 2 * np.pi / num_diffusion_timesteps

    __, pred_e = model(data, t)

    if reduce_mean:
        loss = global_add_pool(e - pred_e, data.batch).square().mean(dim=1)
    else:
        loss = global_add_pool(e - pred_e, data.batch).square().sum(dim=1)

    if not keepdim:
        loss = loss.mean(dim=0)

    return loss


loss_registry = {
    'simple': noise_estimation_loss,
}