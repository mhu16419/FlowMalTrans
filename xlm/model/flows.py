"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"
forward() is the pass from zk to z0 (evaluating)
generate() is the pass from z0 to zk (generating)
"""

import numpy as np
import torch
from torch import nn
import math
import itertools


class MLP(nn.Module):
    """ a simple 4-layer MLP """

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



class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        # s is for scaling
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        # t is for shifting
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, y):
        s = self.s if self.s is not None else y.new_zeros(y.size())
        t = self.t if self.t is not None else y.new_zeros(y.size())
        x = y * torch.exp(s) + t
        # dx / dy
        log_det = torch.sum(s, dim=1)
        return x, log_det

    def generate(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        y = (x - t) * torch.exp(-s)
        # dy / dx
        log_det = torch.sum(-s, dim=1)
        return y, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we cleverly initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    # if reverse=False, for forward
    def initialize_parameters(self, inputs, reverse=False):
        assert self.s is not None and self.t is not None  # for now
        if not reverse:
            self.s.data = (-torch.log(inputs.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(inputs * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True

    def forward(self, y):
        # first batch is used for init
        if not self.data_dep_init_done:
            self.initialize_parameters(y, reverse=False)
        return super().forward(y)

    def generate(self, x):
        return super().generate(x)


class MyActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, dim]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = dim
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, inputs):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(inputs.clone(), dim=0, keepdim=True)
            vars = torch.mean((inputs.clone() + bias) ** 2, dim=0, keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, inputs, reverse=False):
        if reverse:
            return inputs - self.bias
        else:
            return inputs + self.bias

    def _scale(self, inputs, reverse=False):
        if reverse:
            inputs = inputs * torch.exp(-self.logs)
        else:
            inputs = inputs * torch.exp(self.logs)
        logdet = torch.sum(self.logs)
        if reverse:
            logdet *= -1

        return inputs, logdet

    def forward(self, inputs, reverse=False):
        if not self.inited:
            self.initialize_parameters(inputs)

        inputs = self._center(inputs, reverse=False)
        inputs, logdet = self._scale(inputs, reverse=False)

        return inputs, logdet

    def generate(self, inputs):
        inputs, logdet = self._scale(inputs, reverse=True)
        inputs = self._center(inputs, reverse=True)

        return inputs, logdet



class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True, n_layers=3, dropout=0):
        super().__init__()
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
        # print(log_det.sum())
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


class PlanarFlows(nn.Module):

    def __init__(self, dim):
        super(PlanarFlows, self).__init__()
        self.dim = dim
        self.u = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.init_params()

    def init_params(self):
        self.u.data.uniform_(-math.sqrt(1/self.dim), math.sqrt(1/self.dim))
        self.w.data.uniform_(-math.sqrt(1/self.dim), math.sqrt(1/self.dim))
        self.b.data.uniform_(-math.sqrt(1/self.dim), math.sqrt(1/self.dim))

    # the generate pass here is for generating zk based on a sample z0 from base distri.
    def generate(self, z):
        # z: [B, z_size]
        linear_part = torch.mm(z, self.w.T) + self.b
        x = z + self.u * self.h(linear_part)
        # dx / dz
        log_det = self.log_det(z)
        return x, log_det

    def forward(self, x):
        raise NotImplementedError('The forward has not been implemented for Planar FLows.')

    # computing the derivative of tanh
    def h_prime(self, x):
        return 1 - self.h(x) ** 2

    def psi(self, z):
        linear_part = torch.mm(z, self.w.T) + self.b
        return self.h_prime(linear_part) * self.w

    # this is df(z) / dz
    def log_det(self, z):
        inner = 1 + torch.mm(self.psi(z), self.u.T)
        return torch.log(torch.abs(inner)).reshape(-1)


class InversePlanarFlows(PlanarFlows):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.generate = self.generate, self.forward


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        w_init = torch.qr(torch.randn((self.dim, self.dim)))[0]
        p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
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

    # from data space to latent space
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
            raise ValueError('The direction has to be either forward or backward.')

    # from latent space to data space
    def generate(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        ys = [x]
        for flow in self.flows[::-1]:
            x, ld = flow.generate(x)
            log_det += ld
            ys.append(x)
        return ys, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    # from data space x to latent space z
    def forward(self, x):
        zs, log_det = self.flow(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    # from latent space z to data space x
    def generate(self, z):
        xs, log_det = self.flow.generate(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.generate(z)
        return xs


def create_normalizing_flows(flow_type, z_size, dropout, kwargs):
    # flow type
    if flow_type not in ['none', 'planar', 'glow', 'nsf', 'scf']:
        raise ValueError('Error, flow_type %s unknown' % flow_type)

    # then we need parameterize this with normalizing flow
    if flow_type != 'none':
        hiddenflow_flow_nums = kwargs['hiddenflow_flow_nums']
        if flow_type == 'planar':
            # how many flows should be used
            hidden_flows = [PlanarFlows(dim=z_size) for i in range(hiddenflow_flow_nums)]
            flow_net = NormalizingFlow(hidden_flows)
        elif flow_type == 'scf':
            hiddenflow_units = kwargs['hiddenflow_units']
            hiddenflow_layeres = kwargs['hiddenflow_layers']
            flows = [AffineHalfFlow(dim=z_size, parity=i % 2, nh=hiddenflow_units, n_layers=hiddenflow_layeres,
                                    dropout=dropout) for i in range(hiddenflow_flow_nums)]

            flow_net = NormalizingFlow(flows)
        elif flow_type == 'glow':
            hiddenflow_units = kwargs['hiddenflow_units']
            hiddenflow_layeres = kwargs['hiddenflow_layers']
            convs = [Invertible1x1Conv(dim=z_size) for i in range(hiddenflow_flow_nums)]
            norms = [MyActNorm(dim=z_size) for _ in convs]
            couplings = [AffineHalfFlow(dim=z_size, parity=i % 2, nh=hiddenflow_units,
                                        n_layers=hiddenflow_layeres, dropout=dropout) for i in range(len(convs))]
            # append a coupling layer after each 1x1
            hidden_flows = list(itertools.chain(*zip(convs, couplings)))
            flow_net = NormalizingFlow(hidden_flows)
    else:
        flow_net = None

    return flow_net
