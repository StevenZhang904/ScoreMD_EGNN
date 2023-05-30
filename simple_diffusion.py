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
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)

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

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))

    def forward(self, mode, **kwargs):
        if mode == "train":
            # 先判断必须参数
            if "x_start" and "t" in kwargs.keys():
                # 接下来判断一些非必选参数
                if "loss_type" and "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"],
                                             noise=kwargs["noise"], loss_type=kwargs["loss_type"])
                elif "loss_type" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], loss_type=kwargs["loss_type"])
                elif "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], noise=kwargs["noise"])
                else:
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"])

            else:
                raise ValueError("扩散模型在训练时必须传入参数x_start和t！")

        elif mode == "generate":
            if "image_size" and "batch_size" and "channels" in kwargs.keys():
                return self.sample(image_size=kwargs["image_size"],
                                   batch_size=kwargs["batch_size"],
                                   channels=kwargs["channels"])
            else:
                raise ValueError("扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数")
        else:
            raise ValueError("mode参数必须从{train}和{generate}两种模式中选择")

    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)





class MLP(nn.Module):
    def __init__(self, in_size, zDim = 1):
        super(MLP, self).__init__()

        self.enc_mu_FC1 = nn.Linear(in_size, 256)
        self.enc_mu_FC2 = nn.Linear(256, zDim)
        self.enc_var_FC1 = nn.Linear(in_size, 256)
        self.enc_var_FC2 = nn.Linear(256, zDim)
        self.relu = nn.ReLU()

        self.model = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        mu = self.enc_mu_FC1(x)
        mu = self.relu(mu)
        mu = self.enc_mu_FC2(mu)

        logVar = self.enc_var_FC1(x)
        logVar = self.relu(logVar)
        logVar = self.enc_var_FC2(logVar)
        return mu, logVar

    # def forward(self, x):
    #     out = self.model(x)
    #     return self.output(out), out

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # torch.cuda.empty_cache()

    batch_size = 512
    lr_mlp = 0.00005
    lr_resnet = 0.0001

    num_epochs = 15

    transform = transforms.Compose([
        transforms.CenterCrop((380, 380)),
        transforms.Resize((224, 224)),
    ])

    CNN_Data = CNN_Dataset(
        sys_data_dir='/home/cmu/Desktop/Summer_research/ScoreMD_EGNN/sys_data.csv', 
        mem_data_dir='/home/cmu/Desktop/Summer_research/memb_img/', 
        transform= transform
    )

    train_size = int(0.5 * len(CNN_Data))
    test_size = len(CNN_Data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(CNN_Data, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    print('training set size:', len(train_dataset))
    print('testing set size:', len(test_dataset))

    resnet = torchvision.models.resnet18(pretrained = True).to(device)
    for name, p in resnet.named_parameters():
        p.requires_grad = False
    mlp = MLP(in_size=1000).to(device)
    # print(resnet)
    # print(mlp)

    ct = 0
    for child in resnet.children():
        # print("resnet child",child)
        # ct += 1
        # if ct < 7:
        for param in child.parameters():
            param.requires_grad = False

    # loss_func = nn.MSELoss(reduction='mean')
    # loss_func = nn.SmoothL1Loss(reduction='mean')
    optim_mlp = Adam(mlp.parameters(), lr=lr_mlp)
    optim_resnet = Adam(resnet.parameters(), lr=lr_resnet)

    mlp_lr_scheduler = lr_scheduler.MultiStepLR(optim_mlp,gamma=0.5, milestones=[5, 10, 40, 60, 80])
    # resnet_lr_scheduler = lr_scheduler.MultiStepLR(optim_resnet, gamma=0.1, milestones=[400])

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        start = time.time()
        acc_train_loss = 0.0

        resnet.train()
        mlp.train()

        for i, (img, label) in enumerate(train_dataloader):


            img, label = img.to(device), label.to(device)
            if len(label.shape) == 1:
                label = torch.unsqueeze(label, 1)
            
            feat = resnet(img)
            # pred, __ = mlp(feat)

            mu, logVar = mlp(feat)
            std = torch.exp(logVar/2)
            m = torch.distributions.Normal(mu, std)
            # print(mu, mu.shape)
            # print(std, std.shape)
            loss = -m.log_prob(label)
            # print("label", label)
            # print("label shape", label.shape)
            # print(loss, loss.shape)

            # if i == 0:
            #     print(pred[0], label[0])

            optim_mlp.zero_grad()
            # optim_resnet.zero_grad()
            # loss = loss_func(pred, label)
            loss.mean().backward()
            acc_train_loss += loss.mean().item()
            optim_mlp.step()
            # optim_resnet.step()
            torch.cuda.empty_cache()

        train_losses.append(acc_train_loss/(i+1))

        # lr decay
        mlp_lr_scheduler.step()
        # resnet_lr_scheduler.step()
        

        # validation on test data

        # resnet.eval()
        # mlp.eval()
        # predictions = np.zeros(len(test_dataset))
        # labels = np.zeros(len(test_dataset))
        # start_idx, end_idx = 0, 0
        # acc_test_loss = 0.0
        # # with torch.no_grad():

        # for i, (img, label) in enumerate(test_dataloader):
        #     img, label = img.to(device), label.to(device)
        #     if len(label.shape) == 1:
        #         label = torch.unsqueeze(label, 1)
        #     batch_size = label.shape[0]
        #     end_idx += batch_size
            
        #     feat = resnet(img)
        #     pred, __ = mlp(feat)

        #     loss = loss_func(pred, label)
        #     acc_test_loss += loss.item()

        #     if device == 'cpu':
        #         pred = pred.detach().numpy()
        #         label = label.detach().numpy()
        #     else:
        #         pred = pred.detach().cpu().numpy()
        #         label = label.detach().cpu().numpy()

        #     predictions[start_idx:end_idx] = np.squeeze(pred)
        #     labels[start_idx:end_idx] = np.squeeze(label)
        #     start_idx = end_idx
                
        # test_losses.append(acc_test_loss/(i+1))
        # print("epoch: {}, training Loss: {}, testing loss: {}".format(
        #     epoch, train_losses[-1], test_losses[-1]))
        epoch_duration = time.time() - start
        print("epoch: {}, training Loss: {}, epoch time: {}".format(
            epoch, train_losses[-1], epoch_duration))
        # print("epoch: {}, training Loss: {}".format(epoch, train_losses[-1]))

    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(mlp.state_dict(), os.path.join(model_dir, 'mlp_new.ckpt'))
    torch.save(resnet.state_dict(), os.path.join(model_dir, 'resnet_new.ckpt'))
    # torch.save(cnn.state_dict(), os.path.join(model_dir, 'cnn.ckpt'))

    resnet.eval()
    mlp.eval()

    # validation on train data
    # train_predictions = np.zeros(len(train_dataset))
    # train_labels = np.zeros(len(train_dataset))
    # start_idx, end_idx = 0, 0
    # for i, (img, label) in enumerate(train_dataloader):
    #     img, label = img.to(device), label.to(device)
    #     if len(label.shape) == 1:
    #         label = torch.unsqueeze(label, 1)
    #     batch_size = label.shape[0]
    #     end_idx += batch_size
        
    #     feat = resnet(img)
    #     pred, __ = mlp(feat)

    #     if device == 'cpu':
    #         pred = pred.detach().numpy()
    #         label = label.detach().numpy()
    #     else:
    #         pred = pred.detach().cpu().numpy()
    #         label = label.detach().cpu().numpy()

    #     train_predictions[start_idx:end_idx] = np.squeeze(pred)
    #     train_labels[start_idx:end_idx] = np.squeeze(label)
    #     start_idx = end_idx

    # validation on test data
    # test_predictions = np.zeros(len(test_dataset))
    # test_labels = np.zeros(len(test_dataset))
    # start_idx, end_idx = 0, 0
    # for i, (img, label) in enumerate(train_dataloader):
    #     img, label = img.to(device), label.to(device)
    #     if len(label.shape) == 1:
    #         label = torch.unsqueeze(label, 1)
    #     batch_size = label.shape[0]
    #     end_idx += batch_size
    #     print(batch_size, start_idx, end_idx)
        
    #     feat = resnet(img)
    #     pred = mlp(feat)

    #     if device is 'cpu':
    #         pred = pred.detach().numpy()
    #         label = label.detach().numpy()
    #     else:
    #         pred = pred.detach().cpu().numpy()
    #         label = label.detach().cpu().numpy()

    #     test_predictions[start_idx:end_idx] = np.squeeze(pred)
    #     test_labels[start_idx:end_idx] = np.squeeze(label)
    #     start_idx = end_idx

    test_predictions = np.zeros(len(test_dataset))
    test_labels = np.zeros(len(test_dataset))

    start_idx, end_idx = 0, 0
    for i, (img, label) in enumerate(test_dataloader):
        img, label = img.to(device), label.to(device)
        if len(label.shape) == 1:
            label = torch.unsqueeze(label, 1)
        batch_size = label.shape[0]
        end_idx += batch_size

        feat = resnet(img)
        pred, __ = mlp(feat)

        if device == 'cpu':
            pred = pred.detach().numpy()
            label = label.detach().numpy()
        else:
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

        test_predictions[start_idx:end_idx] = np.squeeze(pred)
        test_labels[start_idx:end_idx] = np.squeeze(label)
        start_idx = end_idx

    # print("MSE on training set:", np.mean(np.square(train_predictions - train_labels)))
    print("MSE on testing set:", np.mean(np.square(test_predictions - test_labels)))
    # print("L1 error on training set:", np.mean(np.abs(train_predictions - train_labels)))
    print("L1 error on testing set:", np.mean(np.abs(test_predictions - test_labels)))


    plot_dir = './plot'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    x = np.linspace(np.min(test_dataset.labels), np.max(train_dataset.labels))
    y = np.linspace(np.min(test_dataset.labels), np.max(train_dataset.labels))
    plt.figure()
    plt.scatter(test_predictions, test_labels, c='blue', marker='x')
    # plt.scatter(predictions, labels, c='red', marker='x')
    plt.plot(x, y, linestyle='dashed', c='black')
    plt.xlabel('prediction')
    plt.ylabel('label')
    plt.title('Flux prediction')
    # plt.show()
    plt.savefig(os.path.join(plot_dir, 'pred_new_{}.png'))

    # print(train_losses)
    # print(test_losses)
    plt.figure()
    plt.plot(train_losses, c='blue')
    plt.plot(test_losses, c='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss', 'testing loss'])
    plt.savefig(os.path.join(plot_dir, 'loss_new_{}.png'))


if __name__ == "__main__":
    train()