import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .Graph import get_adjacency_matrix

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

# proposed AGCN

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=-2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1,beta=1):

        N,C,H,W=x.shape
        x1, x2, x3 = self.conv1(x).contiguous().view(N,-1,H*W), self.conv2(x).contiguous().view(N,-1,H*W), self.conv3(x).contiguous().view(N,-1,H*W)
        A1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        A2 = self.soft(x1.unsqueeze(-1) * x2.unsqueeze(-2))
        x1 = self.conv4(A1* alpha+A2* beta)  + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,ncv->ncu', x1, x3).contiguous().view(N,-1,H,W)

        return x1

class unit_ctrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True):
        super(unit_ctrgcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]

        self.convs=CTRGC(in_channels, out_channels)

        self.down = lambda x: x

        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())

        z = self.convs(x, A, self.alpha, self.beta)
        y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y



class CTRGCN(nn.Module):
    def __init__(self, length=14,  in_channels=3, adaptive=True, layer=3):
        super(CTRGCN, self).__init__()

        A = get_adjacency_matrix(length)
        self.layer=layer


        self.gcns = nn.ModuleList()
        for i in range(self.layer):
            self.gcns.append(unit_ctrgcn(in_channels, in_channels, A, adaptive=adaptive))


    def forward(self, x):

        for i in range(self.layer):
            x = self.gcns[i](x)


        return x








