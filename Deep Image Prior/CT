import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from dival import DataPairs
from DEQ_solver.solver import *
import numpy as np
from PIL import Image
from noise import *
from Unet import *
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
from dival import get_standard_dataset
dataset = get_standard_dataset('LoDopab',impl='astra_cuda')
ray_trafo = dataset.get_ray_trafo()
#ray_trafo = dataset.get_ray_trafo()
dataset1 =  DataLoader(dataset.create_torch_dataset())
import odl.contrib.torch.operator as odl_torch
op_layer = odl_torch.OperatorModule(ray_trafo).to('cuda')

#from dival.measure import PSNRMeasure
import json
import os

#hyperparameter
num_iter = 6000
lr=0.01
show_evey =50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_lst = []
PSNR_lst = []
PSNR1_lst = []
#ToTensor = ToTensor()

#network setup
net1 = UnetModel(1,1,8,5,0).to(device)
optimizer = Adam(net1.parameters(), lr=lr,weight_decay=0)
mse = torch.nn.MSELoss().to(device)
#s = sum([np.prod(list(p.size())) for p in net1.parameters()]);
#print ('Number of params: %d' % s)

for ob, gt in dataset1:
    y = ob
    x = gt
    break
ob=y
gt=x

noisy_data = torch.unsqueeze(ob, 1).to(device)
data = torch.unsqueeze(gt, 1).to(device)

#data post-processing

z = add_gaussian_noise(torch.zeros(1,1,362,362),std_dev=1).to(device,dtype=torch.float)

def main(num_iter=3000,rep_psnr=True,show_evey=5,exp_weight=0.99,out_avg = None):
    for i in range(num_iter):
    #        for name, param in net1.named_parameters():
     #           if param.grad is not None:
     #               print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item()}")
            out = net1(z)

            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

            if rep_psnr:
                with torch.no_grad():
                    psnr = PSNR(out, data)
                    psnr1 = PSNR(out_avg, data)
                    PSNR_lst.append(psnr)
                    PSNR1_lst.append(psnr1)

                    if i%show_evey==0:
                        print(f'----------{i}--------')
                        print(psnr)
                        print(psnr1)

            out = op_layer(out)
            loss = mse(out, noisy_data)
            loss_lst.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%show_evey == 0:
                print(loss.item())
            loss_lst.append(loss.item())

if __name__ == '__main__':
    main(num_iter=num_iter,show_evey=show_evey)
    print(max(PSNR_lst))
    x_values = [i for i in range(num_iter)]

    plt.subplot(1, 3, 1)
    plt.plot(x_values, loss_lst, marker=',', linestyle='-', color='b', label='Loss')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Loss Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(x_values,PSNR_lst, label='PSNR', color='r', marker=',', linestyle='-')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('PSNR Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.plot(x_values, PSNR1_lst, label='PSNR', color='g', marker=',', linestyle='-')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('AVG PSNR Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

   # file_name1 = os.path.expanduser('~/Desktop/my_list1.json')

    # Save the list to a JSON file on the desktop
  #  with open(file_name1, 'w') as file:
 #       json.dump(PSNR_lst, file)
