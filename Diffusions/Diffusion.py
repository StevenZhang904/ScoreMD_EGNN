
import torch
from torch import nn
from tqdm import tqdm 
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy


def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素。t是一个包含要检索的索引的张量，
    这些索引对应于a张量中的元素。这个函数的输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0] 
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionModel(nn.Module):
    def __init__(self,timesteps=1000,denoise_model=None, loss_type = "l2"):
        super(DiffusionModel, self).__init__()


        self.denoise_model = denoise_model
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.betas = self.cosine_beta_schedule(timesteps, s = 0.008)
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # forwarddiffusion  
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start.pos)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.pos.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.pos.shape
        )
        # print("sjadijsd", sqrt_alphas_cumprod_t.shape, x_start.pos.shape, sqrt_one_minus_alphas_cumprod_t.shape, noise.shape)

        ### multiplied by 6 to comply with the initialization of t in the simple diffusion.py
        ### both sqrt_one_minus_alphas_cumprod_t and sqrt_alphas_cumprod_t are in shape (batchsize, 1)
        sqrt_one_minus_alphas_cumprod_t_6 = torch.cat((sqrt_one_minus_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t), axis = 1).reshape(len(sqrt_one_minus_alphas_cumprod_t)*6, 1)
        sqrt_alphas_cumprod_t_6 = torch.cat((sqrt_alphas_cumprod_t, sqrt_alphas_cumprod_t, sqrt_alphas_cumprod_t, sqrt_alphas_cumprod_t, sqrt_alphas_cumprod_t, sqrt_alphas_cumprod_t), axis = 1).reshape(len(sqrt_alphas_cumprod_t)*6, 1)
        return sqrt_alphas_cumprod_t_6 * x_start.pos + sqrt_one_minus_alphas_cumprod_t_6 * noise

    def compute_loss(self, x_start, t, loss_type="l2", indicator = 0):
        # print("noise =", noise)
        # print("noise.shape = ", noise.shape)

        noise = torch.randn_like(x_start.pos)
        x_noisy = deepcopy(x_start)
        x_noisy.pos = self.q_sample(x_start=x_start, t=t, noise=noise).to(x_start.pos.device)
        # print("x_start.pos, x_noisy.pos=", x_start.pos, x_noisy.pos)
        # x_noisy shape = (batch_size , 18)
        # print("x_noisy= ", x_noisy, "its shape = ", x_noisy.shape)
        predicted_noise = self.denoise_model(x_noisy.x, x_noisy.pos, x_noisy.batch, t).reshape(noise.shape)

        if indicator == 1:
                
            print()
            print("noise shape = ,", noise)
            print("predicted_noise shape = ", predicted_noise)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, timesteps):
        device = next(self.denoise_model.parameters()).device

        b = shape[0] ### b = batch_size
        # print(shape)
        # start from pure noise (for each example in the batch)
        displacement = torch.randn(shape, device=device)
        # start from the mean of positions
        # displacement = [[2.0273046, 1.9712732, 9.537093, 2.0279317, 1.9765526, 9.552533, 2.0245256, 1.9664096, 9.546699,
        #                  2.027873, 1.9693326, 9.437345, 2.0301685, 1.9696349, 9.445996, 2.0288558, 1.9688513, 9.447838]  for x in range(b)]
        # displacement = torch.tensor(displacement, device = device)
        displacements = []

        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=self.timesteps):
            displacement = self.p_sample(displacement, torch.full((b,), i, device=device, dtype=torch.long), i)
            displacements.append(displacement.cpu().numpy())
        return displacements

    @torch.no_grad()
    def sample(self, displacement_shape, timesteps, batch_size=16):
        return self.p_sample_loop(shape=(batch_size, displacement_shape), timesteps= timesteps)

    def forward(self, mode, t = None, x_start = None, displacement_shape = None, batch_size = 256, timesteps_sample = 1000, indicator = 0):
        if mode == "train":
            return self.compute_loss(x_start, t, self.loss_type, indicator=indicator)
        elif mode == "generate":
            print("generating")
            return self.sample(displacement_shape=displacement_shape, batch_size=batch_size, timesteps= timesteps_sample)
        else:
            raise NotImplementedError


    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)