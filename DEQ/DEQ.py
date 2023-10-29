from DEQ_solver.solver import *
import torch
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
from noise import *
from Unet import *
import matplotlib.pyplot as plt
import json
import os
from torch.autograd import gradcheck

#from dival.measure import PSNRMeasure

#hyperparameter
num_iter = 2000
lr=0.01
show_evey =5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_lst = []
PSNR_lst = []
ToTensor = ToTensor()

#network setup
net1 = UnetModel(3,3,8,5,0)
mse = torch.nn.MSELoss().to(device)
#s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
#print ('Number of params: %d' % s)

#data post-processing
pic = r'D:\pythonProject\dataset\train\F16_GT.png'
image = Image.open(pic)
data = ToTensor(image.copy())
data = torch.unsqueeze(data,0)
noisy_data = add_gaussian_noise(data,std_dev=25).to(device,dtype=torch.float)
z = add_gaussian_noise(torch.zeros(1,3,512,512),std_dev=1).to(device)
#z = DataLoader(z,batch_size=1)
deqmodel = DEQResModel(3,3,64,128,0).to('cuda')
#gradcheck(deqmodel, torch.randn(1,3,512,512).double().requires_grad_().to(device,dtype=torch.float), eps=1e-5, atol=1e-3, check_undefined_grad=False)
optimizer = Adam(deqmodel.parameters(),lr=lr,weight_decay=0)
def main(num_iter=3000,rep_psnr=True,show_evey=50):
    for i in range(num_iter):
            print(f'----------{i}--------')

        #    for k in z:
    #        for name, param in net1.named_parameters():
     #           if param.grad is not None:
     #               print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item()}")
            out = deqmodel(z)
            if rep_psnr:
                with torch.no_grad():
                    psnr = PSNR(out.detach().cpu(),data)
                    if i%show_evey == 0:
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
    main(num_iter=num_iter,show_evey=show_evey)
    print(max(PSNR_lst))
    x_values = [i for i in range(num_iter)]

    plt.subplot(1, 2, 1)
    plt.plot(x_values, loss_lst, marker=',', linestyle='-', color='b', label='Data Points')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Loss Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_values,PSNR_lst, label='Data 2', color='r', marker=',', linestyle='-')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('PSNR Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    file_name1 = os.path.expanduser('~/Desktop/my_list.json')

    # Save the list to a JSON file on the desktop
    with open(file_name1, 'w') as file:
        json.dump(PSNR_lst, file)