import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from Unet import *
import matplotlib.pyplot as plt

#torch.__version__
def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].view_as(x0), res

def forward_iteration(f, x0, max_iter=100, tol=1e-2,**kwargs):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res

class DEQFixedPoint(nn.Module):
    """ This was taken from the Deep Equilibrium tutorial here: http://implicit-layers-tutorial.org/deep_equilibrium_models/"""
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z

def tv1_loss(x):
    #### here, our input must be {batch, channel, height, width}
    ndims = len(list(x.size()))
    if ndims != 4:
        assert False, "Input must be {batch, channel, height, width}"
    n_pixels = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    tot_var = torch.sum(dh) + torch.sum(dw)
    tot_var = tot_var / n_pixels
    return tot_var


class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

class DEQResModel(nn.Module):
    def __init__(self,in_chans, out_chans,n_channels, n_inner_channels,drop_prob):
        super(DEQResModel, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.n_channels = n_channels
        self.n_inner_channels = n_inner_channels
        self.drop_prob = drop_prob
        self.f = ResNetLayer(self.n_channels,self.n_inner_channels)
        self.model = nn.Sequential(
            ConvBlock(self.in_chans,self.n_channels,self.drop_prob),
            DEQFixedPoint(self.f, forward_iteration, tol=1e-2, max_iter=100, m=5),
            nn.Conv2d(64,64,kernel_size=2,padding=0,stride=2,bias=False),
            TransposeConvBlock(self.n_channels,self.out_chans),
        )

    def forward(self,input):
        return self.model(input)


def PSNR(original, compressed):
    # Ensure that the images are in the same shape
    if original.size() != compressed.size():
        raise ValueError("Image shapes must match for PSNR calculation.")

    # Calculate the Mean Squared Error (MSE)
    mse = F.mse_loss(compressed, original)

    # Calculate the maximum possible pixel value
    max_pixel_value = 1.0  # Assuming pixel values are in the range [0, 1]

    # Calculate the PSNR
    psnr = 10 * torch.log10((max_pixel_value**2) / mse)

    return psnr.item()

if __name__ == '__main__':
    deqmodel = DEQResModel(3,3,64,128,0).to('cuda')
  #  gradcheck(deq, torch.randn(1, 64, 512, 512).double().requires_grad_(), eps=1e-5, atol=1e-3, check_undefined_grad=False)
