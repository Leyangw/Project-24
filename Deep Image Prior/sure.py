import torch
from torch.optim import Adam
import numpy as np
from PIL import Image
from noise import *
from Unet import *
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import random
import torchvision
from dival.measure import PSNRMeasure
from E_SUREloss import *

ToTensor = ToTensor()
PSNRm = PSNRMeasure(short_name='a', data_range=1)
# seed_everything(7)

num_iter = 20000
lr = 0.008
show_evey = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_lst = []
PSNR_lst = []
PSNR1_lst = []
# network setup
#unet = True
#from skip import *

pad = 'reflection'
input_depth = 3
#net2 = UnetModel(1, 1, 8, 5, 0).to(device)
net1 = UnetModel(3, 3, 4, 5, 0).to(device)
net2 = UnetModel(3, 3, 4, 5, 0).to(device)
# net2 = skip(
#               input_depth, 3,
#              num_channels_down = [4,8,16,32,64],
#             num_channels_up   = [4,8,16,32,64],
#            num_channels_skip = [4,6, 6, 6, 6],
#           upsample_mode='bilinear',
#          need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').to(device)
# if unet:
#	net1 =skip(
#               input_depth, 3,
#              num_channels_down = [4,8,16, 32, 64],
#             num_channels_up   = [4,8,16, 32, 64],
#            num_channels_skip = [4, 4, 4, 4, 4],
#           upsample_mode='bilinear',
#          need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').to(device)
# else:
#	net2 = DnCNN(channels=3).to(device)
# net3 = skip(
#               input_depth, 3,
#              num_channels_down = [128, 128, 128, 128, 128],
#             num_channels_up   = [128, 128, 128, 128, 128],
#            num_channels_skip = [4, 8, 8, 8, 8],
#           upsample_mode='bilinear',
#          need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').to(device)
mse = torch.nn.MSELoss().to(device)
s = sum([np.prod(list(p.size())) for p in net2.parameters()]);
print('Number of params: %d' % s)
#print('kodim')
# data post-processing
pic = r'D:\pythonProject\dataset\train\F16_GT.png'
image = Image.open(pic)
data = ToTensor(image.copy())
resize = torchvision.transforms.Resize([256, 256])
data = resize(torch.unsqueeze(data, 0))
sigma = 25
noisy_data = add_gaussian_noise(data, std_dev=sigma).to(device, dtype=torch.float)
optimizer = Adam(list(net1.parameters()) + list(net2.parameters()), lr=lr, weight_decay=0)
z0 = add_gaussian_noise(torch.zeros(noisy_data.size()).to(device), std_dev=1)
zl = torch.zeros(noisy_data.size()).to(device)

def main(num_iter=3000, rep_psnr=True, show_evey=50, exp_weight=0.99, out_avg=None, noise1=None):
    for i in range(num_iter):
        noise = add_gaussian_noise(zl, std_dev=0.5)

        if noise1 == None:
            noise1 = noise
            z1 = z0 + noise1
        else:
            noise1 = 0.8 * noise1 + 0.2 * noise
            z1 = z0 + noise1

        out1 = net1(z1.requires_grad_())
        angle1 = random.randint(1, 359)
        noisya1 = torchvision.transforms.functional.rotate(noisy_data, angle=angle1)

        input1 = torch.clamp(out1+noise1, -1, 1)

        out2 = net2(input1)
        out3 = net2(torchvision.transforms.functional.rotate(input1, angle=angle1))
        if i == 0:
            outk = out2.detach()
        else:
            outk_1 = outk.detach()
            outk = out2.detach()
        if out_avg is None:
            out_avg = out2
        else:
            out_avg = out_avg * exp_weight + out2.detach() * (1 - exp_weight)

        if rep_psnr:
            with torch.no_grad():
                psnr = PSNRm(out1.detach().cpu(), data)
                psnr1 = PSNRm(out_avg.detach().cpu(), data)
                psnr3 = PSNRm(input1.detach().cpu(), data)
                psnr4 = PSNRm(out2.detach().cpu(), data)
                PSNR_lst.append(psnr4)
                PSNR1_lst.append(psnr1)
        if i >= 0:
         #   zz = z1.requires_grad_()
            divergence = 2 * divergence_new(z1,out2) -sigma
            loss1 =torch.sum( mse(out3, noisya1) + divergence)
            loss2 = mse(out2, noisy_data)
            loss3 = divergence_new(z1, out2)
           # loss3 = 0.4 * mse(outref, noisy_data)
            loss4 = 0.4 * mse(out1, noisy_data)
           # loss5 = 0.15 * mse(out4, noisya2)
          #  loss5 = 0.05 * tv1_loss(out2)
         #   loss7 = 0.1 * mse(out1, outref.detach())
            lossfn = loss1 #+ loss2  + loss4 #+ loss5

        optimizer.zero_grad()
        lossfn.sum().backward(retain_graph=True)
        optimizer.step()

        if i % show_evey == 0:
            print(f'----------{i}--------')
            print(psnr, psnr1, psnr3, lossfn.item())
        loss_lst.append(lossfn.item())


if __name__ == '__main__':
    main(num_iter=num_iter, show_evey=show_evey)
    print(max(PSNR_lst))
    print(max(PSNR1_lst))
