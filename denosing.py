import torch
from torch.optim import Adam
from torchvision.transforms import ToTensor

import numpy as np
from PIL import Image
from noise import *
from Unet import *

from dival.measure import PSNRMeasure

#hyperparameter
num_iter = 7000
lr=0.1
show_evey =100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_lst = []
PSNR_lst = []
ToTensor = ToTensor()

#network setup
net1 = UnetModel(3,3,8,1,0).to(device)
optimizer = Adam(net1.parameters(), lr=lr,weight_decay=0)
mse = torch.nn.MSELoss().to(device)
#s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
#print ('Number of params: %d' % s)

#data post-processing
pic = r'D:\pythonProject\dataset\train\F16_GT.png'
image = Image.open(pic)
data = ToTensor(image.copy())
data = torch.unsqueeze(data,0)
noisy_data = add_gaussian_noise(data,std_dev=25).to(device,dtype=torch.float)
z = add_gaussian_noise(torch.zeros(1,3,512,512),std_dev=1).to(device,dtype=torch.float)

go = True

def main(num_iter=3000,rep_psnr=True):
    for i in range(num_iter):
        print(f'----------{i}--------')
        for name, param in net1.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item()}")
        out = net1(z)

        if rep_psnr:
            with torch.no_grad():
                psnr = PSNRMeasure(short_name=f'psnr_{i}').apply(out.detach().cpu(),data)
                print(psnr)
                PSNR_lst.append(psnr)
                
        lossfn = mse(out,noisy_data)
      #  print(loss.item())
    #    loss_lst.append(loss.item())

        optimizer.zero_grad()
        lossfn.backward()
        optimizer.step()

        print(lossfn.item())
        loss_lst.append(lossfn.item())
        if i ==100:
           break

if __name__ == '__main__':
    main()





