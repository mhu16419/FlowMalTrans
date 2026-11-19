"""
Normalizing flows used by the flow adapter.

The building blocks (MLP, affine half / coupling flow, invertible 1x1 convolution, flow container) follow the
normalizing-flow code of Andrej Karpathy, https://github.com/karpathy/pytorch-normalizing-flows .

A flow is a stack of invertible layers acting on the sentence-level latent code z
(shape [batch, z_size]):
    forward(y):   data space -> base space (standard Gaussian), returns the intermediate
                  tensors and the log |det J|, used for the density-estimation (NLL) loss
    generate(x):  base space -> data space (the inverse pass)
Notice that generate() is the inversion of forward(), not backprop.

Two flow types are supported (--flow_type):
    scf   a stack of RealNVP-style affine coupling layers (AffineHalfFlow)
    glow  Glow-style blocks: an invertible 1x1 convolution followed by an affine coupling layer
"""

import itertools

import torch
from torch import nn


FLOW_TYPES = ('none', 'scf', 'glow')


class MLP(nn.Module):
    """ a simple MLP: `n_layers` hidden layers (Linear + LeakyReLU) and a final linear layer """

    def __init__(self, nin, nout, nh, n_layers=3, dropout=0):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(nin, nh))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2))
            nin = nh
        final_layer = nn.Linear(nh, nout)
        self.reset_parameters(final_layer)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def reset_parameters(self, module):
        init_range = 0.07
        module.weight.data.uniform_(-init_range, init_range)
        module.bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine coupling layer (x = (y - t) * exp(-s)), where half of the
    dimensions in y are linearly scaled/transformed as a function of the other half.
    Which half is which is determined by the parity bit. `dim` must be even.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True, n_layers=3, dropout=0):
        super().__init__()
        assert dim % 2 == 0, 'the coupling layers need an even z_size, got %i' % dim
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        # two networks, one for scale and one for shift
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh, n_layers=n_layers, dropout=dropout)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh, n_layers=n_layers, dropout=dropout)

    def forward(self, y):
        y0, y1 = y[:, :self.dim // 2], y[:, self.dim // 2:]  # this separate the first half and second half dimensions
        if self.parity:
            y0, y1 = y1, y0
        s = self.s_cond(y0)
        t = self.t_cond(y0)
        x0 = y0  # untouched half
        x1 = (y1 - t) * torch.exp(-s)  # transform this half as a function of the other
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        # dx / dy
        log_det = torch.sum(-s, dim=1)
        return x, log_det

    def generate(self, x):
        x0, x1 = x[:, :self.dim // 2], x[:, self.dim // 2:]  # this separate the first half and second half dimensions
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        y0 = x0  # this was the same
        y1 = x1 * torch.exp(s) + t  # reverse the transform on this half
        if self.parity:
            y0, y1 = y1, y0
        y = torch.cat([y0, y1], dim=1)
        # dy / dx
        log_det = torch.sum(s, dim=1)
        return y, log_det


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        w_init = torch.linalg.qr(torch.randn((self.dim, self.dim)))[0]
        p, lower, upper = torch.linalg.lu(w_init)
        s = torch.diag(upper)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        upper = torch.triu(upper, 1)
        l_mask = torch.tril(torch.ones((self.dim, self.dim)), -1)
        eye = torch.eye(self.dim)

        self.register_buffer("p", p)
        self.register_buffer("sign_s", sign_s)
        self.lower = nn.Parameter(lower)
        self.log_s = nn.Parameter(log_s)
        self.upper = nn.Parameter(upper)
        self.l_mask = l_mask
        self.eye = eye

    def get_weight(self, input, reverse):
        self.l_mask = self.l_mask.to(input.device)
        self.eye = self.eye.to(input.device)
        lower = self.lower * self.l_mask + self.eye
        u = self.upper * self.l_mask.transpose(0, 1).contiguous()
        u += torch.diag(self.sign_s * torch.exp(self.log_s))

        dlogdet = torch.sum(self.log_s)
        if reverse:
            u_inv = torch.inverse(u)
            l_inv = torch.inverse(lower)
            p_inv = torch.inverse(self.p)
            weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
        else:
            weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight, dlogdet

    def forward(self, y):
        weight, log_det = self.get_weight(y, reverse=False)
        x = y @ weight
        # dx / dy
        return x, log_det

    def generate(self, x):
        weight, log_det = self.get_weight(x, reverse=True)
        y = x @ weight
        # dy / dx
        return y, -log_det


# ------------------------------------------------------------------------

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    # from data space to base space (direction='fwd'), or the inverse (direction='gen')
    def forward(self, y, direction='fwd'):
        m, _ = y.shape
        log_det = torch.zeros(m).to(y.device)
        if direction == 'fwd':
            xs = [y]
            for flow in self.flows:
                y, ld = flow(y)
                log_det += ld
                xs.append(y)
            return xs, log_det
        elif direction == 'gen':
            x = y
            ys = [x]
            for flow in self.flows[::-1]:
                x, ld = flow.generate(x)
                log_det += ld
                ys.append(x)
            return ys, log_det
        else:
            raise ValueError("The direction has to be either 'fwd' or 'gen'.")

    # from base space to data space
    def generate(self, x):
        return self.forward(x, direction='gen')


def create_normalizing_flows(flow_type, z_size, dropout, kwargs):
    """
    Build a NormalizingFlow of type `flow_type` ('scf' or 'glow') for latent codes of size
    `z_size`, or return None for 'none'. `kwargs` holds hiddenflow_flow_nums (number of
    flow layers), hiddenflow_units (hidden size of the coupling MLPs) and hiddenflow_layers
    (number of hidden layers of the coupling MLPs).
    """
    flow_type = flow_type.lower()
    if flow_type not in FLOW_TYPES:
        raise ValueError('Error, flow_type %s unknown (use one of: %s)' % (flow_type, ', '.join(FLOW_TYPES)))
    if flow_type == 'none':
        return None

    n_flows = kwargs['hiddenflow_flow_nums']
    units = kwargs['hiddenflow_units']
    n_layers = kwargs['hiddenflow_layers']
    if flow_type == 'scf':
        flows = [AffineHalfFlow(dim=z_size, parity=i % 2, nh=units, n_layers=n_layers, dropout=dropout)
                 for i in range(n_flows)]
    else:  # glow
        convs = [Invertible1x1Conv(dim=z_size) for _ in range(n_flows)]
        couplings = [AffineHalfFlow(dim=z_size, parity=i % 2, nh=units, n_layers=n_layers, dropout=dropout)
                     for i in range(n_flows)]
        # append a coupling layer after each 1x1
        flows = list(itertools.chain(*zip(convs, couplings)))
    return NormalizingFlow(flows)
