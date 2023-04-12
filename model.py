import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils import spectral_norm


class GLU(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(input_channels, output_channels * 2, 1, 1, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x1, x2 = x.chunk(2, dim=1)
        out = self.sigmoid(x1) * x2
        return out

class SkipLayerExcitation(nn.Module):
    def __init__(self, hr_channels, lr_channels):
        super().__init__()
        self.adapool = nn.AdaptiveAvgPool2d(4)
        self.c1 = spectral_norm(nn.Conv2d(lr_channels, lr_channels, 4, 1, 0, bias=False))
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.c2 = spectral_norm(nn.Conv2d(lr_channels, hr_channels, 1, 1, 0, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, hr, lr):
        lr = self.adapool(lr)
        lr = self.c1(lr)
        lr = self.leaky_relu(lr)
        lr = self.c2(lr)
        lr = self.sigmoid(lr)
        hr = hr * lr
        return hr


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x + self.gain * torch.randn(x.shape, device=x.device)
        return x


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(input_channels, output_channels * 4, 1, 1, 0, bias=False))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=1, upsample=False):
        super().__init__()
        if num_layers == 1:
            self.seq = nn.Sequential(
                    NoiseInjection(),
                    spectral_norm(nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False)),
                    nn.LeakyReLU(0.1))
        else:
            self.seq = nn.Sequential()
            for _ in range(num_layers):
                self.seq.append(NoiseInjection())
                self.seq.append(
                        spectral_norm(nn.Conv2d(input_channels, input_channels, 3, 1, 1, bias=False)))
                self.seq.append(nn.LeakyReLU(0.1))
            self.seq.append(
                    spectral_norm(nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False)))
        if upsample:
            self.seq.append(Upsample(output_channels, output_channels))

    def forward(self, x):
        return self.seq(x)


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        return x


class Generator(nn.Module):
    def __init__(self,
                 latent_dim=256,
                 sle_map=[(0, 4), (1, 6), (2, 8)],
                 num_blocks=9,
                 num_layers_per_block=1,
                 upsample_layers = [0, 1, 2, 3, 5, 7],
                 channels = [1024, 512, 256, 128, 64, 64, 64, 32, 16, 8],
                 grayscale_output_layer = 4,
                 output_channels = 3,
                 norm_class = ChannelNorm
                 ):
        super().__init__()
        channels = channels.copy()

        self.first_layer = nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(latent_dim, latent_dim, 4, 1, 0)),
                norm_class(latent_dim),
                GLU(latent_dim, latent_dim),
                ConvBlock(latent_dim, latent_dim, num_layers=num_layers_per_block), 
                nn.Upsample(scale_factor=(2, 2)),
                spectral_norm(nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)),
                norm_class(latent_dim),
                GLU(latent_dim, channels[0]),
                )
        self.mid_layers = nn.ModuleList([])
        self.sles = nn.ModuleList([])
        for i in range(num_blocks):
            if i+1 == num_blocks:
                c_next = channels[-1]
            else:
                c_next = channels[i+1]
            c = channels[i]
            upsample_flag = (i in upsample_layers)
            self.mid_layers.append(
                    ConvBlock(c, c_next, num_layers=num_layers_per_block, upsample=upsample_flag))
            for lr, hr in sle_map:
                if lr == i:
                    self.sles.append(SkipLayerExcitation(channels[hr+1], channels[lr+1]))
        self.sle_map = sle_map
        self.grayscale_output_layer_id = grayscale_output_layer
        self.grayscale_output_layer = nn.Sequential(
                spectral_norm(nn.Conv2d(channels[self.grayscale_output_layer_id+1], 1, 3, 1, 1, bias=False)),
                nn.Tanh()
                )

        self.last_layer = nn.Sequential(
                spectral_norm(nn.Conv2d(channels[-1], output_channels, 1, 1, 0, bias=False)),
                nn.Tanh()
                )

    def forward(self, x):
        x = self.first_layer(x)
        grayscale = None
        outputs = []
        for i, layer in enumerate(self.mid_layers):
            x = layer(x)
            outputs.append(x)
            for j, (lr, hr) in enumerate(self.sle_map):
                if hr == i:
                    x = self.sles[j](x, outputs[lr])
            if i == self.grayscale_output_layer_id:
                grayscale = self.grayscale_output_layer(x)
        x = self.last_layer(x)
        return x, grayscale


class ProjectedSubdiscriminator(nn.Module):
    def __init__(self, internal_channels, num_downsamples):
        super().__init__()
        self.downsamples = nn.Sequential(*[
            nn.Sequential(nn.LazyConv2d(internal_channels, 4, 2, 0), nn.LeakyReLU(0.1))
            for _ in range(num_downsamples)])
        self.output_layer = nn.Sequential(
                nn.LazyConv2d(64, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.LazyConv2d(1, 3, 1, 1),
                )

    def forward(self, x):
        x = self.downsamples(x)
        x = torch.cat([x, x.std(dim=0, keepdim=True).repeat(x.shape[0], 1, 1, 1)], dim=1)
        x = self.output_layer(x)
        return x


class ProjectedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        self.projectors = nn.ModuleList([nn.LazyConv2d(64 * (2 ** i), 1, 1, 0) for i in range(4)])
        for param in self.projectors.parameters():
            param.resuires_grad = False

        self.subdiscriminators = nn.ModuleList([
                ProjectedSubdiscriminator(64, 3),
                ProjectedSubdiscriminator(128, 2),
                ProjectedSubdiscriminator(256, 1),
                ProjectedSubdiscriminator(256, 0),
            ])

        self.csms = nn.ModuleList([
                nn.Conv2d(512, 256, 1, 1, 0, bias=False),
                nn.Conv2d(256, 128, 1, 1, 0, bias=False),
                nn.Conv2d(128, 64, 1, 1, 0, bias=False)
            ])
        for param in self.csms.parameters():
            param.requires_grad = False

    def forward(self, x):
        projected = []
        for i, layer in enumerate(self.efficientnet.features[:-1]):
            x = layer(x)
            if i in [2, 3, 4, 6]:
                projected.append(self.projectors[[2, 3, 4, 6].index(i)](x))
        projected[2] += F.interpolate(self.csms[0](projected[3]), scale_factor=2)
        projected[1] += F.interpolate(self.csms[1](projected[2]), scale_factor=2)
        projected[0] += F.interpolate(self.csms[2](projected[1]), scale_factor=2)
        logits = 0 
        for i, (sd, p) in enumerate(zip(self.subdiscriminators, projected)):
            logits += (sd(p)).mean(dim=(2, 3))
        return logits


# Discriminator for  Low Resolution (128x Grayscale) 
class LowResolutionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
                spectral_norm(nn.Conv2d(1, 64, 3, 1, 1)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Conv2d(64, 64, 4, 2, 0)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Conv2d(64, 64, 3, 1, 1)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Conv2d(64, 64, 4, 2, 0)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Conv2d(64, 64, 3, 1, 1)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Conv2d(64, 64, 4, 2, 0)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Conv2d(64, 64, 3, 1, 1)),
                nn.LeakyReLU(0.1),
                spectral_norm(nn.Conv2d(64, 64, 4, 2, 0)),
                )

    def forward(self, x):
        return self.seq(x).mean(dim=(2, 3))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_disc = ProjectedDiscriminator()
        self.grayscale_disc = LowResolutionDiscriminator()

    def forward(self, x, gs):
        return self.grayscale_disc(gs), self.proj_disc(x)
