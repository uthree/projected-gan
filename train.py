import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from dataset import ImageDataset
from model import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path')
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-e', '--num-epoch', default=100, type=int)
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)

args = parser.parse_args()
device = torch.device(args.device)

G = Generator().to(device)
D = Discriminator().to(device)

if os.path.exists('./generator.pt'):
    G.load_state_dict(torch.load('./generator.pt', map_location=device))
if os.path.exists('./discriminator.pt'):
    D.load_state_dict(torch.load('./discriminator.pt', map_location=device))

ds = ImageDataset([args.dataset_path], max_len=args.max_data, size=512)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

OptG = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
OptD = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

to_grayscale = nn.Sequential(
        T.Resize((128, 128)),
        T.Grayscale()
        )

for epoch in range(args.num_epoch):
    bar = tqdm(total=len(ds))
    for batch, real in enumerate(dl):
        N = real.shape[0]
        real = real.to(device)
        real_grayscale = to_grayscale(real).detach()
        
        # Train G.
        z = torch.randn(N, 256, 1, 1).to(device)
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            fake, fake_grayscale = G(z)
            logit = D(fake, fake_grayscale)
            g_loss = ((logit - 1) ** 2).mean()
        scaler.scale(g_loss).backward()
        scaler.step(OptG)
        
        # Train D.
        fake = fake.detach()
        fake_grayscale = fake_grayscale.detach()
        
        OptD.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logit_real = D(real, real_grayscale)
            logit_fake = D(fake, fake_grayscale)
            d_loss = ((logit_real - 1) ** 2).mean() + ((logit_fake + 1) ** 2).mean()
        scaler.scale(d_loss).backward()
        scaler.step(OptD)
        bar.set_description(desc=f"G: {g_loss.item():.4f}, D: {d_loss.item():.4f}")
        bar.update(N)

        if batch % 100 == 0:
            torch.save(G.state_dict(), './generator.pt')
            torch.save(D.state_dict(), './discriminator.pt')
            torchvision.io.write_jpeg(((fake[0].cpu() * 127.5) + 127.5).to(torch.uint8), 'preview.jpg')
            torchvision.io.write_jpeg(((fake_grayscale[0].cpu() * 127.5) + 127.5).to(torch.uint8), 'preview_gs.jpg')


        scaler.update()
