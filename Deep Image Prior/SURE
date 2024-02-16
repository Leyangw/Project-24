import torch
import torch.nn as nn

def divergence_new(net_input, out):
    b_prime = torch.randn_like(net_input).type(torch.float)
    nh_y = torch.sum(b_prime * out, dim=[1, 2, 3])
    vector = torch.ones(1).to(out)
    divergence = b_prime * torch.autograd.grad(nh_y, net_input, grad_outputs=vector, retain_graph=True, create_graph=True)[0]
    return divergence
