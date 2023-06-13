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
from ..dataset.data_diffusion_demo import Diffusion_Dataset
from .Diffusion import DiffusionModel
import time 
import math



class SinusoidalPositionEmbeddings(nn.Module):
    '''
    The SinusoidalPositionEmbeddings module takes a tensor of shape (batch_size, 1) 
    as input (i.e. the noise levels of several noisy images in a batch), 
    and turns this into a tensor of shape (batch_size, dim), 
    with dim being the dimensionality of the position embeddings. 
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1) 
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # print(time[:, None])
        # print(embeddings[None, :])
        embeddings = (time[:, None] * embeddings[None, :]).squeeze(1)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Denoise_model(nn.Module):
    def __init__(self, in_size, out_size = 18, time_mlp_in_size = 8):
        super(Denoise_model, self).__init__()
        self.time_mlp_in_size = time_mlp_in_size
        self.in_size = in_size 
        self.out_size = out_size

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_mlp_in_size),
            nn.Linear(self.time_mlp_in_size, self.time_mlp_in_size*4),
            nn.GELU(),
            nn.Linear(self.time_mlp_in_size*4, self.time_mlp_in_size*4),
        )
        self.model = nn.Sequential(
            nn.Linear(self.in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.output = nn.Linear(256+self.time_mlp_in_size*4, self.out_size)


     
    def forward(self, x, t ):
        time = self.time_mlp(t)
        res = self.model(x)
        x = torch.cat((res, time), -1)
        res = self.output(x)
        return res

    # def forward(self, x):
    #     out = self.model(x)
    #     return self.output(out), out

def train():
    # torch.cuda.empty_cache()

    batch_size = 1120


    Diffusion_Data = Diffusion_Dataset(
        data_dir = "/home/cmu/Desktop/Summer_research/position_data_2.csv",
        sys_dir = "/home/cmu/Desktop/Summer_research/sys_pdb/",
        name = "circular_22_",
    )
    train_size = int(0.8 * len(Diffusion_Data))
    test_size = len(Diffusion_Data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(Diffusion_Data, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=7)
    print('training set size:', len(train_dataset))
    print('testing set size:', len(test_dataset))
    
    # hidden_dim = 1024
    # zdim = 40
    # encoder = Encoder(18, hidden_dim, zdim)
    # decoder = Decoder(zdim, hidden_dim, 18)
    train_losses, test_losses = [], []

    timesteps = 7000
    epoches = 10000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_path = "/home/cmu/Desktop/Summer_research/ScoreMD_EGNN/"
    # denoise_model = VAE(encoder, decoder).to(device)


    # denoise_model = Unet(
    #     dim=image_size,
    #     channels=1,
    #     dim_mults=dim_mults
    # )
    time_mlp_in_size = 20
    denoise_model = Denoise_model(in_size = 18, out_size= 18, time_mlp_in_size = time_mlp_in_size).to(device)
    lr = 0.00001

    optimizer = Adam(denoise_model.parameters(), lr=lr)
    noise_scale = 0.5
    model = DiffusionModel(timesteps=timesteps, denoise_model = denoise_model, loss_type= "l2").to(device)

    # denoise_model.load_state_dict(torch.load("/home/cmu/Desktop/Summer_research/ScoreMD_EGNN/BestModel_MLP.pth"))
    # model.load_state_dict(torch.load("/home/cmu/Desktop/Summer_research/ScoreMD_EGNN/BestModel_Diff.pth"))

    for i in tqdm(range(epoches)):
        losses = []
        loss_temp = math.inf
        for step, (features, labels) in enumerate(train_dataloader):
            features = features.to(device)
            batch_size = features.shape[0]

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long() # shape = [batch_size]

            loss = model(mode="train", x_start=features, t=t, noise_scale = noise_scale)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 100 == 0:

            print("In epoch ", i, ", loss is ", sum(losses)/len(losses))
            if loss_temp > sum(losses)/len(losses):
                torch.save(model.state_dict(), model_save_path+'BestModel_Diff_emd_dim_16.pth')
                torch.save(denoise_model.state_dict(), model_save_path+'BestModel_MLP_emd_dim_16.pth')
                print("model saved to" + str(model_save_path))
                loss_temp = sum(losses)/len(losses)

    with torch.no_grad():
        model.eval()
        ### so far, rule of thumb of this sampling timestamps is around 4000
        timesteps_sample = 4000
        samples = model(mode="generate", t=t, displacement_shape = 18, batch_size = 4, timesteps_sample = timesteps_sample)
        torch.set_printoptions(precision = 4, sci_mode=False)
        print("time = ", timesteps_sample, "result =" , samples[-1]*3.535067+4.4975667)
        

if __name__ == "__main__":
    train()