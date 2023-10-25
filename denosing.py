import torch
from torch.optim import Adam

import numpy as np
from PIL import Image
import pandas
import os
from noise import *
from Unet import *

from dival.measure import PSNRMeasure

pic = r'D:\pythonProject\dataset\train\F16_GT.png'
image = Image.open(pic)
num_iter = 7000
lr=0.01
show_evey =100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rep_psnr = True
loss_lst = []
PSNR_lst = []

#s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
#print ('Number of params: %d' % s)
data = np.asarray(image)
data = np.transpose(data, (2, 0, 1))
data = data.copy()
noisy_data = get_noisy_image(data,25)
noisy_data = torch.from_numpy(noisy_data)

net = UnetModel(3,3,8,4,0).to(device)
optimizer = Adam(list(net.parameters()), lr=lr, weight_decay=0)
mse = torch.nn.MSELoss()

data = torch.from_numpy(data)
data = torch.unsqueeze(data,0)
noisy_data = torch.unsqueeze(noisy_data,0).to(device)
z = torch.from_numpy(get_noisy_image(np.zeros((1,3,512,512)),1)).to(device)
if __name__ == '__main__':
    for i in range(num_iter):
        print(f'----------{i}--------')
  #      for name, param in net.named_parameters():
     #       if param.grad is not None:
    #            print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item()}")
        out = net(z)

        if rep_psnr:
            with torch.no_grad():
                psnr = PSNRMeasure(short_name=f'psnr_{i}').apply(out.detach().cpu(),data)
                print(psnr)
                PSNR_lst.append(psnr)
                
        loss = mse(out,noisy_data).to(device)
        print(loss.item())
        loss_lst.append(loss.item())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()







