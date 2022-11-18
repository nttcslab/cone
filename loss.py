import torch
import torch.nn as nn
import torch.nn.functional as F


class loss_t(nn.Module):
    def __init__(self, omega_f=1.5):
        super(loss_t, self).__init__()
        self.omega_f = omega_f
        self.SmoothnessLoss = SmoothnessLoss()
        self.FidelityLoss = nn.MSELoss()

    def forward(self, x, t):
        omega_f = self.omega_f
        L_sm = self.SmoothnessLoss(x, t)
        L_f = self.FidelityLoss(x, t)
        L_t = L_sm + omega_f * L_f
        return L_t


class loss_y(nn.Module):
    def __init__(self, omega_c=0.5):
        super(loss_y, self).__init__()
        self.omega_c = omega_c
        self.ExposureControlLoss = ExposureControlLoss(16, 0.6)
        self.SpatialConsistencyLoss = SpatialConsistencyLoss(4)
        self.ColorConstancyLoss = ColorConstancyLoss()

    def forward(self, x, y):
        omega_c = self.omega_c
        L_e = self.ExposureControlLoss(y)
        L_sp = self.SpatialConsistencyLoss(x, y)
        L_c = self.ColorConstancyLoss(y)
        L_y = L_e + L_sp + omega_c * L_c
        return L_y


class SmoothnessLoss(nn.Module):
    def __init__(self, sigma=10, p=1.0):
        super(SmoothnessLoss, self).__init__()
        self.sigma = sigma
        self.p = p # p-norm

    def rgb2ycbcr(self, RGB):
        RGB2 = RGB.contiguous().view(-1, 3).float()
        weight = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        YCBCR = RGB2.mm(weight) + bias
        YCBCR = YCBCR.view(RGB.shape[0], 3, RGB.shape[2], RGB.shape[3])
        return YCBCR

    def forward(self, x, t):
        sigma = self.sigma
        p = self.p
        x = self.rgb2ycbcr(x)
        gamma = -1.0 / (2 * sigma * sigma)

        w01 = torch.exp(torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=1,
                                  keepdim=True) * gamma)
        w02 = torch.exp(torch.sum(torch.pow(x[:, :, :-1, :] - x[:, :, 1:, :], 2), dim=1,
                                  keepdim=True) * gamma)
        w03 = torch.exp(torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=1,
                                  keepdim=True) * gamma)
        w04 = torch.exp(torch.sum(torch.pow(x[:, :, :, :-1] - x[:, :, :, 1:], 2), dim=1,
                                  keepdim=True) * gamma)
        w05 = torch.exp(torch.sum(torch.pow(x[:, :, :-1, :-1] - x[:, :, 1:, 1:], 2), dim=1,
                                  keepdim=True) * gamma)
        w06 = torch.exp(torch.sum(torch.pow(x[:, :, 1:, 1:] - x[:, :, :-1, :-1], 2), dim=1,
                                  keepdim=True) * gamma)
        w07 = torch.exp(torch.sum(torch.pow(x[:, :, 1:, :-1] - x[:, :, :-1, 1:], 2), dim=1,
                                  keepdim=True) * gamma)
        w08 = torch.exp(torch.sum(torch.pow(x[:, :, :-1, 1:] - x[:, :, 1:, :-1], 2), dim=1,
                                  keepdim=True) * gamma)
        w09 = torch.exp(torch.sum(torch.pow(x[:, :, 2:, :] - x[:, :, :-2, :], 2), dim=1,
                                  keepdim=True) * gamma)
        w10 = torch.exp(torch.sum(torch.pow(x[:, :, :-2, :] - x[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * gamma)
        w11 = torch.exp(torch.sum(torch.pow(x[:, :, :, 2:] - x[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * gamma)
        w12 = torch.exp(torch.sum(torch.pow(x[:, :, :, :-2] - x[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * gamma)
        w13 = torch.exp(torch.sum(torch.pow(x[:, :, :-2, :-1] - x[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * gamma)
        w14 = torch.exp(torch.sum(torch.pow(x[:, :, 2:, 1:] - x[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * gamma)
        w15 = torch.exp(torch.sum(torch.pow(x[:, :, 2:, :-1] - x[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * gamma)
        w16 = torch.exp(torch.sum(torch.pow(x[:, :, :-2, 1:] - x[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * gamma)
        w17 = torch.exp(torch.sum(torch.pow(x[:, :, :-1, :-2] - x[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * gamma)
        w18 = torch.exp(torch.sum(torch.pow(x[:, :, 1:, 2:] - x[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * gamma)
        w19 = torch.exp(torch.sum(torch.pow(x[:, :, 1:, :-2] - x[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * gamma)
        w20 = torch.exp(torch.sum(torch.pow(x[:, :, :-1, 2:] - x[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * gamma)
        w21 = torch.exp(torch.sum(torch.pow(x[:, :, :-2, :-2] - x[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * gamma)
        w22 = torch.exp(torch.sum(torch.pow(x[:, :, 2:, 2:] - x[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * gamma)
        w23 = torch.exp(torch.sum(torch.pow(x[:, :, 2:, :-2] - x[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * gamma)
        w24 = torch.exp(torch.sum(torch.pow(x[:, :, :-2, 2:] - x[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * gamma)

        # g: gradient
        g01 = w01 * torch.norm((t[:, :, 1:, :] - t[:, :, :-1, :]), p, dim=1, keepdim=True)
        g02 = w02 * torch.norm((t[:, :, :-1, :] - t[:, :, 1:, :]), p, dim=1, keepdim=True)
        g03 = w03 * torch.norm((t[:, :, :, 1:] - t[:, :, :, :-1]), p, dim=1, keepdim=True)
        g04 = w04 * torch.norm((t[:, :, :, :-1] - t[:, :, :, 1:]), p, dim=1, keepdim=True)
        g05 = w05 * torch.norm((t[:, :, :-1, :-1] - t[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        g06 = w06 * torch.norm((t[:, :, 1:, 1:] - t[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        g07 = w07 * torch.norm((t[:, :, 1:, :-1] - t[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        g08 = w08 * torch.norm((t[:, :, :-1, 1:] - t[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        g09 = w09 * torch.norm((t[:, :, 2:, :] - t[:, :, :-2, :]), p, dim=1, keepdim=True)
        g10 = w10 * torch.norm((t[:, :, :-2, :] - t[:, :, 2:, :]), p, dim=1, keepdim=True)
        g11 = w11 * torch.norm((t[:, :, :, 2:] - t[:, :, :, :-2]), p, dim=1, keepdim=True)
        g12 = w12 * torch.norm((t[:, :, :, :-2] - t[:, :, :, 2:]), p, dim=1, keepdim=True)
        g13 = w13 * torch.norm((t[:, :, :-2, :-1] - t[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        g14 = w14 * torch.norm((t[:, :, 2:, 1:] - t[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        g15 = w15 * torch.norm((t[:, :, 2:, :-1] - t[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        g16 = w16 * torch.norm((t[:, :, :-2, 1:] - t[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        g17 = w17 * torch.norm((t[:, :, :-1, :-2] - t[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        g18 = w18 * torch.norm((t[:, :, 1:, 2:] - t[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        g19 = w19 * torch.norm((t[:, :, 1:, :-2] - t[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        g20 = w20 * torch.norm((t[:, :, :-1, 2:] - t[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        g21 = w21 * torch.norm((t[:, :, :-2, :-2] - t[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        g22 = w22 * torch.norm((t[:, :, 2:, 2:] - t[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        g23 = w23 * torch.norm((t[:, :, 2:, :-2] - t[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        g24 = w24 * torch.norm((t[:, :, :-2, 2:] - t[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        L_sm = torch.mean(g01) \
               + torch.mean(g02) \
               + torch.mean(g03) \
               + torch.mean(g04) \
               + torch.mean(g05) \
               + torch.mean(g06) \
               + torch.mean(g07) \
               + torch.mean(g08) \
               + torch.mean(g09) \
               + torch.mean(g10) \
               + torch.mean(g11) \
               + torch.mean(g12) \
               + torch.mean(g13) \
               + torch.mean(g14) \
               + torch.mean(g15) \
               + torch.mean(g16) \
               + torch.mean(g17) \
               + torch.mean(g18) \
               + torch.mean(g19) \
               + torch.mean(g20) \
               + torch.mean(g21) \
               + torch.mean(g22) \
               + torch.mean(g23) \
               + torch.mean(g24)
        return L_sm


class ExposureControlLoss(nn.Module):
    def __init__(self, kernel_size, epsilon):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size)
        self.epsilon = epsilon

    def forward(self, y):
        # b, c, h, w = y.shape
        y = torch.mean(y, 1, keepdim=True)
        y = self.pool(y)
        epsilon = torch.FloatTensor([self.epsilon]).cuda()
        L_e = torch.mean(torch.pow(y - epsilon, 2))
        return L_e


class SpatialConsistencyLoss(nn.Module):
    def __init__(self, size):
        super(SpatialConsistencyLoss, self).__init__()
        self.pool = nn.AvgPool2d(size)
        kernel_l = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0) # left
        kernel_r = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0) # right
        kernel_u = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0) # up
        kernel_d = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0) # down
        self.weight_l = nn.Parameter(data=kernel_l, requires_grad=False)
        self.weight_r = nn.Parameter(data=kernel_r, requires_grad=False)
        self.weight_u = nn.Parameter(data=kernel_u, requires_grad=False)
        self.weight_d = nn.Parameter(data=kernel_d, requires_grad=False)

    def forward(self, x, y):
        # b, c, h, w = y.shape
        x = torch.mean(x, 1, keepdim=True)
        y = torch.mean(y, 1, keepdim=True)

        x = self.pool(x)
        y = self.pool(y)

        # D: derivative
        Dx_l = F.conv2d(x, self.weight_l, padding=1)
        Dx_r = F.conv2d(x, self.weight_r, padding=1)
        Dx_u = F.conv2d(x, self.weight_u, padding=1)
        Dx_d = F.conv2d(x, self.weight_d, padding=1)

        Dy_l = F.conv2d(y, self.weight_l, padding=1)
        Dy_r = F.conv2d(y, self.weight_r, padding=1)
        Dy_u = F.conv2d(y, self.weight_u, padding=1)
        Dy_d = F.conv2d(y, self.weight_d, padding=1)

        # se: squared error
        se_l = torch.pow(Dx_l - Dy_l, 2)
        se_r = torch.pow(Dx_r - Dy_r, 2)
        se_u = torch.pow(Dx_u - Dy_u, 2)
        se_d = torch.pow(Dx_d - Dy_d, 2)

        # mse
        L_sp = torch.mean(se_l + se_r + se_u + se_d)
        return L_sp


class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, y):
        # b, c, h, w = y.shape
        y = torch.mean(y, [2, 3], keepdim=True)
        y_R, y_G, y_B = torch.split(y, 1, dim=1)

        # se: squared error
        se_RG = torch.pow(y_R - y_G, 2)
        se_RB = torch.pow(y_R - y_B, 2)
        se_GB = torch.pow(y_G - y_B, 2)
        
        L_c = torch.mean(se_RG + se_RB + se_GB)
        return L_c
