import torch
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from DEQ_solver.solver import *
from torch.autograd import gradcheck
import numpy as np
from PIL import Image
from noise import *
from Unet import *
import matplotlib.pyplot as plt
import random
from E_SUREloss import *
from dival.measure import PSNRMeasure
PSNRm = PSNRMeasure(short_name='a',data_range=1)
import json
import os
seed_everything(1)
#hyperparameter
num_iter = 5500
lr=0.01
show_evey =100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_lst = []
PSNR_lst = []
PSNR1_lst=[]
ToTensor = ToTensor()

#network setup
net1 = UnetModel(3,3,4,5,0).to(device)
mse = torch.nn.MSELoss().to(device)
s = sum([np.prod(list(p.size())) for p in net1.parameters()]);
print ('Number of params: %d' % s)

#data post-processing
pic = r'D:\pythonProject\dataset\train\F16_GT.png'
image = Image.open(pic)
data = ToTensor(image.copy())
data = torch.unsqueeze(data,0)
noisy_data = add_gaussian_noise(data,std_dev=25).to(device,dtype=torch.float)

optimizer = Adam(net1.parameters(), lr=lr,weight_decay=0)

z0 = add_gaussian_noise(torch.zeros(1,3,512,512).to(device),std_dev=1).requires_grad_()
#zl= torch.zeros(1,3,512,512).to(device)

def main(num_iter=3000,rep_psnr=True,show_evey=5,exp_weight=0.99,out_avg = None,z1=None,noise1 = None):
    for i in range(num_iter):
        input = add_gaussian_noise(noisy_data,std_dev=1).to(device,dtype=torch.float).requires_grad_()
        out = net1(input)

        if out_avg is None:
            out_avg = out
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        if rep_psnr:
            with torch.no_grad():
                psnr = PSNRm(out.detach().cpu(),data)
                psnr1=PSNRm(out_avg.detach().cpu(),data)
                if i%show_evey == 0:
                    print(f'----------{i}--------')
                    print(psnr)
                    print(psnr1)
                PSNR_lst.append(psnr)
                PSNR1_lst.append(psnr1)
        divergence = divergence_new(input, out)
        lossfn = torch.sum(mse(out,noisy_data)+ 2 * divergence*25 -25)
  #      print(divergence.item())
      #      loss1 = mse(out3, noisya1) + divergence

        optimizer.zero_grad()
        lossfn.backward()
        optimizer.step()

        if i%show_evey == 0:
            print(lossfn.item())
        loss_lst.append(lossfn.item())

if __name__ == '__main__':
    main(num_iter=num_iter,show_evey=show_evey)
    print(max(PSNR_lst))
    print(max(PSNR1_lst))
    print(PSNR1_lst)
    x_values = [i for i in range(num_iter)]


