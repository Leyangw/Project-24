import torch
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
from noise import *
from Unet import *
import matplotlib.pyplot as plt

from dival.measure import PSNRMeasure

#hyperparameter
num_iter = 4000
lr=0.1
show_evey =50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_lst = []
PSNR_lst = []
ToTensor = ToTensor()

#network setup
net1 = UnetModel(3,3,8,5,0).to(device)
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
z = DataLoader(z,batch_size=1)

def main(num_iter=3000,rep_psnr=True):
    for i in range(num_iter):
        for k in z:
    #        for name, param in net1.named_parameters():
     #           if param.grad is not None:
     #               print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item()}")
            out = net1(k)

            if rep_psnr and i%show_evey == 0:
                print(f'----------{i}--------')
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

            if i%show_evey == 0:
                print(lossfn.item())
            loss_lst.append(lossfn.item())

if __name__ == '__main__':
    main(num_iter=num_iter)
    x_values = [i for i in range(num_iter)]
    plt.plot(x_values, loss_lst, marker='o', linestyle='-', color='b', label='Data Points')
    plt.plot(PSNR_lst, label='Data 2', color='r', marker='s', linestyle='--')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()





