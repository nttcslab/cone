import loss
import torch
import torch.nn as nn


eps = 0.0001


# The 'baseline' directly regards the reflectance 'r' as the
# desired enhancement result 'y', similar to convenitonal
# illumination estimation-centric methods.
class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()

    def forward(self, x, t):
        y = x / t
        y = torch.clamp(y, 0, 1)
        return y


class BetaGammaCorrection(nn.Module):
    def __init__(self):
        super(BetaGammaCorrection, self).__init__()
        self.a = nn.Parameter(torch.ones(1) * (-0.3293))
        self.b = nn.Parameter(torch.ones(1) * 1.1258)

    def forward(self, x, t):
        a = self.a
        b = self.b
        k = 1 / t
        beta = torch.exp(b * (1 - torch.pow(k, a)))
        gamma = torch.pow(k, a)
        y = beta * torch.pow(x, gamma)
        y = torch.clamp(y, 0, 1)
        return y


class PreferredCorrection(nn.Module):
    def __init__(self):
        super(PreferredCorrection, self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, x, t):
        a = torch.clamp(self.a, min=eps)
        b = torch.clamp(self.b, min=eps)
        k = 1 / t
        numerator = torch.pow(k, a * b) * x
        denominator = torch.pow((torch.pow(k, a) - 1) * torch.pow(x, 1 / b) + 1, b)
        y = numerator / denominator
        y = torch.clamp(y, 0, 1)
        return y


class SigmoidCorrection(nn.Module):
    def __init__(self):
        super(SigmoidCorrection, self).__init__()

        # For the sigmoid correction, we want to initialize the parameter 'b'
        # in our paper as inf, but this may cause the gradient to be too large
        # and clipped by 'torch.nn.utils.clip_grad_norm_'. To avoid this
        # problem, we define 'b' here as the inverse of 'b' in our paper and
        # initialize it as 0. Consequently, the comparametric equation coded
        # here differs from that in our paper.
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1) * 0)

    def forward(self, x, t):
        a = torch.clamp(self.a, min=eps)
        b = self.b
        k = 1 / t
        numerator = (b + 1) * torch.pow(k, a) * x
        denominator = b * (torch.pow(k, a) - 1) * x + (b + 1)
        y = numerator / denominator
        y = torch.clamp(y, 0, 1)
        return y


class EnhancementNetwork(nn.Module):
    def __init__(self, blocks, channels):
        super(EnhancementNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(blocks):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=kernel_size, stride=1, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        t = x + fea
        t = torch.clamp(t, eps, 1)
        return t


class SelfCalibratedNetwork(nn.Module):
    def __init__(self, blocks, channels):
        super(SelfCalibratedNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(blocks):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=kernel_size, stride=1, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, r):
        fea = self.in_conv(r)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        delta = r - fea
        return delta


class network(nn.Module):
    def __init__(self, stages=3, cem='sigmoid'):
        super(network, self).__init__()
        self.stages = stages
        self.enhance = EnhancementNetwork(blocks=1, channels=3)
        self.calibrate = SelfCalibratedNetwork(blocks=3, channels=16)
        
        cem_dic = {'baseline': baseline(),
                   'betagamma': BetaGammaCorrection(),
                   'preferred': PreferredCorrection(),
                   'sigmoid': SigmoidCorrection()}
        self.cem = cem_dic[cem]
        
        self.L_t = loss.loss_t()
        self.L_y = loss.loss_y()

    def weights_init(self, net):
        if isinstance(net, nn.Conv2d):
            net.weight.data.normal_(0., 0.02)
            net.bias.data.zero_()
        if isinstance(net, nn.BatchNorm2d):
            net.weight.data.normal_(1., 0.02)

    def forward(self, x):
        x_list = []
        t_list = []
        x2 = x
        for i in range(self.stages):
            x_list.append(x2)
            t = self.enhance(x2)
            r = x / t
            r = torch.clamp(r, 0, 1)
            delta = self.calibrate(r)
            x2 = x + delta
            t_list.append(t)

        t = t_list[0]
        y = self.cem(x, t)
        return x_list, t_list, y

    def _loss(self, x):
        x_list, t_list, y = self(x)
        L = 0
        for i in range(self.stages):
            L += self.L_t(x_list[i], t_list[i])
        L += self.L_y(x, y)
        return L


class inference(nn.Module):
    def __init__(self, path, cem='sigmoid'):
        super(inference, self).__init__()
        self.enhance = EnhancementNetwork(blocks=1, channels=3)

        cem_dic = {'baseline': baseline(),
                   'betagamma': BetaGammaCorrection(),
                   'preferred': PreferredCorrection(),
                   'sigmoid': SigmoidCorrection()}
        self.cem = cem_dic[cem]
        
        self.load_state_dict(torch.load(path), strict=False)

    def forward(self, x):
        t = self.enhance(x)
        y = self.cem(x, t)
        return y
