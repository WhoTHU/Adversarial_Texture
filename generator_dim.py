import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Generator_dis(nn.Module):
    def __init__(self, DIM=128, z_dim=16, img_shape=(32, 32)):
        super(Generator_dis, self).__init__()
        self.DIM = DIM
        # self.final_shape = (np.array(img_shape) / 32).astype(np.int64)
        self.final_shape = [9, 9]
        self.final_dim = np.prod(self.final_shape)
        # self.img_shape  = self.final_shape * 32
        self.img_shape = img_shape
        preprocess = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, 1, 1),
            nn.BatchNorm2d(z_dim),
            nn.Tanh(),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 4, stride=2, padding=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(2 * DIM),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(2 * DIM, 4 * DIM, 1, stride=1, padding=0),
            # nn.BatchNorm2d(4 * DIM),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(4 * DIM, 2 * DIM, 3, stride=1, padding=1),
            # nn.BatchNorm2d(2 * DIM),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(2 * DIM, 4 * DIM, 1, stride=1, padding=0),
            # nn.BatchNorm2d(4 * DIM),
            # nn.ReLU(True),

            # nn.ConvTranspose2d(4 * DIM, 2 * DIM, 4, stride=2, padding=3),
            # nn.BatchNorm2d(2 * DIM),
            # nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # deconv_out = nn.ConvTranspose2d(DIM, self.cl_num, 4, stride=2, padding=3)
        deconv_out = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=3)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        # self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        # output = self.tanh(output)
        output = F.sigmoid(output)
        return output


class Discriminator_dis(nn.Module):
    def __init__(self, DIM=128, img_shape=(324, 324)):
        super(Discriminator_dis, self).__init__()

        self.DIM = DIM
        # self.final_shape = (np.array(img_shape) / 32).astype(np.int64)
        self.final_shape = [9, 9]
        self.final_dim = np.prod(self.final_shape)
        # self.img_shape  = self.final_shape * 32
        self.img_shape = img_shape
        self.main = nn.Sequential(
            # nn.Conv2d(self.cl_num, DIM, 3, 2, padding=3),
            nn.Conv2d(3, DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 2 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 4, 2, padding=3),
            nn.LeakyReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(5 * DIM, DIM, 1, 1),
            nn.LeakyReLU(),
        )

        self.linear = nn.Sequential(nn.Linear(self.final_dim * DIM, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 1),
                                    )

    def forward(self, input, z):
        output = self.main(input)
        output = torch.cat([output, z], dim=1)
        output = self.block2(output)
        output = output.view(output.shape[0], self.final_dim * self.DIM)
        output = self.linear(output)
        return output


class GAN_dis(nn.Module):
    def __init__(self, DIM=128, z_dim=16, img_shape=(324, 324)):
        super(GAN_dis, self).__init__()
        self.DIM = DIM
        self.z_dim = z_dim
        # self.final_shape = (np.array(img_shape) / 32).astype(np.int64)
        self.final_shape = [9, 9]
        self.final_dim = np.prod(self.final_shape)
        # self.img_shape = self.final_shape * 32
        self.img_shape = img_shape
        self.G = Generator_dis(self.DIM, self.z_dim, self.img_shape)
        self.D = Discriminator_dis(self.DIM, self.img_shape)

    def forward(self, input):
        pass

    def get_loss(self, y, z, gp=0.0):
        z_prime = torch.cat((z[1:], z[:1]), dim=0)
        pj = self.D(y, z)
        pm = self.D(y, z_prime)
        Ej = -F.softplus(-pj).mean()
        Em = F.softplus(pm).mean()
        loss = (Em - Ej)
        if gp > 0:
            return loss + gp * calc_gradient_penalty(self.D, y, z), pj, pm
        else:
            return loss, pj, pm

    def generate(self, z=None, batch_size=None, sample_mode=None):
        # z = self.z
        if z is None:
            z = torch.randn(batch_size, self.DIM, self.final_shape[0], self.final_shape[1])
            z = z.to(self.D.linear.weight)

        x = self.G(z)
        x_proj = x
        return x_proj

    def discriminate(self, x):
        return self.D(x)

    def anneal(self, steps=1):
        self.temp = max(self.temp * np.exp(- steps * self.anneal_rate), self.temp_min)

def calc_gradient_penalty(net, real_data, real_data2):

    # alpha = torch.rand(real_data.shape[0], 1, 1, 1)
    # # alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    # # alpha = alpha.view(batch_size, 3, dim, dim)
    # alpha = alpha.to(real_data)
    #
    # interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    #
    # interpolates.requires_grad_(True)

    interpolates = real_data.clone().detach().requires_grad_()
    interpolates2 = real_data2.clone().detach().requires_grad_()
    disc_interpolates = net(interpolates, interpolates2)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=[interpolates, interpolates2],
                                    grad_outputs=torch.ones(disc_interpolates.size(), device=disc_interpolates.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = [g.view(g.shape[0], -1) for g in gradients]
    gradient_penalty = 0.0
    for g in gradients:
        gradient_penalty = gradient_penalty + ((g.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
