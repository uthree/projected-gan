import argparse
import os
import torch
import torchvision
from tqdm import tqdm

from model import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-n', '--num-images', default=1, type=int)

args = parser.parse_args()
device = torch.device(args.device)

G = Generator().to(device)
D = Discriminator().to(device)

if os.path.exists('./generator.pt'):
    G.load_state_dict(torch.load('./generator.pt', map_location=device))
if os.path.exists('./discriminator.pt'):
    D.load_state_dict(torch.load('./discriminator.pt', map_location=device))

if not os.path.exists('./results'):
    os.mkdir('./results')

for i in tqdm(range(args.num_images)):
    z = torch.randn(1, 256, 1, 1).to(device)
    fake, _ = G(z)
    torchvision.io.write_jpeg(((fake[0].cpu() * 127.5) + 127.5).to(torch.uint8),
            os.path.join('./results/', f"{i}.jpg"))
    
