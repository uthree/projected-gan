import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Blur(nn.Module):
    def __init__(self):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x


class SkipLayerExcitation(nn.Module):
    def __init__(self, hr_channels, lr_channels):
        super().__init__()
        self.adapool = nn.AdaptiveAvgPool2d(4)
        self.c1 = nn.Conv2d(lr_channels, lr_channels, 4, 1, 0)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.c2 = nn.Conv2d(lr_channels, hr_channels, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hr, lr):
        lr = self.adapool(lr)
        lr = self.c1(lr)
        lr = self.leaky_relu(lr)
        lr = self.c2(lr)
        lr = self.sigmoid(lr)
        hr = hr * lr
        return hr


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels * 4, 1, 1, 0)
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
                    nn.Conv2d(input_channels, output_channels, 3, 1, 1),
                    Blur(),
                    nn.LeakyReLU(0.1))
        else:
            self.seq = nn.Sequential()
            for _ in range(num_layers):
                self.seq.append(nn.Conv2d(input_channels, input_channels, 3, 1, 1))
                self.seq.append(nn.LeakyReLU(0.1))
            self.seq.append(nn.Conv2d(input_channels, output_channels, 3, 1, 1,))
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
                 sle_map=[(0, 3), (1, 5), (2, 7)],
                 num_blocks=9,
                 num_layers_per_block=1,
                 upsample_layers = [0, 1, 2, 3, 5, 7],
                 channels = [64, 64, 64, 64, 64, 64, 3, 3],
                 grayscale_output_layer = 4,
                 output_channels = 3,
                 norm_class = nn.BatchNorm2d
                 ):
        super().__init__()
        channels = channels.copy()
        channels.insert(0, latent_dim // 4)

        self.first_layer = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, latent_dim, 4, 1, 0),
                norm_class(latent_dim),
                nn.GLU(dim=1),
                ConvBlock(latent_dim // 2, latent_dim // 2, num_layers=num_layers_per_block), 
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(latent_dim // 2, latent_dim // 2, 3, 1, 1),
                norm_class(latent_dim // 2),
                nn.GLU(dim=1)
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
                    self.sles.append(SkipLayerExcitation(channels[hr+1], channels[lr]))
        self.sle_map = sle_map
        self.grayscale_output_layer_id = grayscale_output_layer
        self.grayscale_output_layer = nn.Sequential(
                nn.Conv2d(channels[self.grayscale_output_layer_id+1], 1, 3, 1, 1),
                nn.Tanh()
                )

        self.last_layer = nn.Sequential(
                nn.Conv2d(channels[-1], output_channels, 3, 1, 1),
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


class ProjectedDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        self.projectors = nn.ModuleList([nn.LazyConv2d(64, 1, 1, 0, bias=False) for _ in range(8)])
        for param in self.projectors.parameters():
            param.requires_grad = False
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.LazyConv2d(64, 1, 1, 0),
                nn.LeakyReLU(0.1),
                nn.LazyConv2d(1, 1, 1, 0)) for _ in range(8)])
        self.last_discriminator = nn.Sequential(
                nn.LazyConv2d(64, 1, 1, 0),
                nn.LeakyReLU(0.1),
                nn.LazyConv2d(1, 1, 1, 0))

    def forward(self, x):
        projected = []
        logits = 0
        for i, layer in enumerate(self.efficientnet.features[:-1]):
            x = layer(x)
            projected.append(self.projectors[i](x))
        for i, p in enumerate(projected):
            logits = logits + self.discriminators[i](p).mean(dim=(2, 3))
        mb_std = x.std(dim=0, keepdim=True).mean(dim=(1, 2, 3), keepdim=True)
        x = x + mb_std
        logits = logits + self.last_discriminator(x).mean()
        return logits


# Discriminator for  Low Resolution (128x Grayscale) 
class LowResolutionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, 4, 2, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, 4, 2, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, 4, 2, 2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64, 1, 4, 2, 2),
                )

    def forward(self, x):
        return self.seq(x).mean(dim=(2, 3))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_disc = ProjectedDiscriminator()
        self.grayscale_disc = LowResolutionDiscriminator()

    def forward(self, x, gs):
        return self.proj_disc(x) + self.grayscale_disc(gs)
