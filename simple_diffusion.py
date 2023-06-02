import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from dataset.data_diffusion_demo import Diffusion_Dataset
import time 
import math

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
    def __init__(self,timesteps=1000,denoise_model=None):
        super(DiffusionModel, self).__init__()


        self.denoise_model = denoise_model

        # 方差生成
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
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # forwarddiffusion  (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l2"):
        # print("noise =", noise)
        # print("noise.shape = ", noise.shape)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print("x_noisy= ", x_noisy, "its shape = ", x_noisy.shape)
        predicted_noise = self.denoise_model(x_noisy, t)
        # print("predicted_noise = ,", predicted_noise, "its shape = ", predicted_noise.shape)

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
        # Use our model (noise predictor) to predict the mean
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
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device

        b = shape[0] ### b = batch_size
        # start from pure noise (for each example in the batch)
        displacement = torch.randn(shape, device=device)
        displacements = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            displacement = self.p_sample(displacement, torch.full((b,), i, device=device, dtype=torch.long), i)
            displacements.append(displacement.cpu().numpy())
        return displacements

    @torch.no_grad()
    def sample(self, displacement_shape, batch_size=16):
        return self.p_sample_loop(shape=(batch_size, displacement_shape))

    # def forward(self, mode, **kwargs):
    #     if mode == "train":
    #         # 先判断必须参数
    #         if "x_start" and "t" in kwargs.keys():
    #             # 接下来判断一些非必选参数
    #             if "loss_type" and "noise" in kwargs.keys():
    #                 return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"],
    #                                          noise=kwargs["noise"], loss_type=kwargs["loss_type"])
    #             elif "loss_type" in kwargs.keys():
    #                 return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], loss_type=kwargs["loss_type"])
    #             elif "noise" in kwargs.keys():
    #                 return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], noise=kwargs["noise"])
    #             else:
    #                 return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"])

    #         else:
    #             raise ValueError("扩散模型在训练时必须传入参数x_start和t！")

    #     elif mode == "generate":
    #         if "image_size" and "batch_size" and "channels" in kwargs.keys():
    #             return self.sample(image_size=kwargs["image_size"],
    #                                batch_size=kwargs["batch_size"],
    #                                channels=kwargs["channels"])
    #         else:
    #             raise ValueError("扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数")
    #     else:
    #         raise ValueError("mode参数必须从{train}和{generate}两种模式中选择")
    def forward(self, mode, x_start, t):
        if mode == "train":
            loss_type = "l2"
            t = t
            noise = torch.randn_like(x_start)
            return self.compute_loss(x_start, t, noise, loss_type)
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

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var


class Decoder(nn.Module):
        ''' This the decoder part of VAE

        '''
        def __init__(self, z_dim, hidden_dim, output_dim):
            '''
            Args:
                z_dim: A integer indicating the latent size.
                hidden_dim: A integer indicating the size of hidden dimension.
                output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
            '''
            super().__init__()

            self.linear = nn.Linear(z_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is of shape [batch_size, latent_dim]

            hidden = F.relu(self.linear(x))
            # hidden is of shape [batch_size, hidden_dim]

            predicted = torch.sigmoid(self.out(hidden))
            # predicted is of shape [batch_size, output_dim]

            return predicted


class VAE(nn.Module):
        ''' This the VAE, which takes a encoder and decoder.

        '''
        def __init__(self, enc, dec):
            super().__init__()

            # self.in_size = in_size
            # self.out_size = out_size

            # self.time_mlp = nn.Sequential(
            #     SinusoidalPositionEmbeddings(in_size),
            #     nn.Linear(in_size, in_size*4),
            #     nn.GELU(),
            #     nn.Linear(in_size*4, in_size*4),
            # )
            # self.t = self.time_mlp(time)
            self.enc = enc
            self.dec = dec

        def forward(self, x, t=None):
            # encode

            z_mu, z_var = self.enc(x)

            # sample from the distribution having latent parameters z_mu, z_var
            # reparameterize
            std = torch.exp(z_var / 2)
            eps = torch.randn_like(std)
            x_sample = eps.mul(std).add_(z_mu)

            # decode
            predicted = self.dec(x_sample)
            # return predicted, z_mu, z_var
            return predicted

class Denoise_model(nn.Module):
    def __init__(self, in_size, out_size = 18, time = None):
        super(Denoise_model, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(in_size),
        #     nn.Linear(in_size, in_size*4),
        #     nn.GELU(),
        #     nn.Linear(in_size*4, in_size*4),
        # )
        # t = self.time_mlp(time)

        # self.model = nn.Sequential(
        #     nn.Linear(self.in_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        # )
        # self.output = nn.Linear(512, self.out_size)


     
    def forward(self, x):
        res = self.model(x)
        res = self.output(res)
        return res

    # def forward(self, x):
    #     out = self.model(x)
    #     return self.output(out), out

def train():
    # torch.cuda.empty_cache()

    batch_size = 256


    Diffusion_Data = Diffusion_Dataset(
        data_dir = "/home/cmu/Desktop/Summer_research/position_data_2.csv",
        sys_dir = "/home/cmu/Desktop/Summer_research/sys_pdb/",
        name = "circular_22_",
    )
    train_size = int(0.8 * len(Diffusion_Data))
    test_size = len(Diffusion_Data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(Diffusion_Data, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('training set size:', len(train_dataset))
    print('testing set size:', len(test_dataset))

    encoder = Encoder(18, 256, 20)
    decoder = Decoder(20, 256, 18)
    train_losses, test_losses = [], []

    timesteps = 1000
    epoches = 100
    train_loader = train_dataloader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_path = "/home/cmu/Desktop/Summer_research/ScoreMD_EGNN/"
    denoise_model = VAE(encoder, decoder).to(device)
    lr = 0.0001
    optimizer = Adam(denoise_model.parameters(), lr=lr)
    model = DiffusionModel(timesteps=timesteps, denoise_model= denoise_model)


    for i in range(epoches):
        losses = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (features, labels) in loop:
            features = features.to(device)
            batch_size = features.shape[0]

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = model(mode="train", x_start=features, t=t)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新信息
            loop.set_description(f'Epoch [{i}/{epoches}]')
            loop.set_postfix(loss=loss.item())
        print("In epoch ", i, ", loss is ", losses[0])
        torch.save(model.state_dict(), model_save_path+'BestModel.pth')
        print("model saved to" + str(model_save_path))


if __name__ == "__main__":
    train()