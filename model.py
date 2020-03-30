import sys

from torch.distributions.normal import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 bn=False,
                 num_groups=8):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        identity = x
        res = self.relu(x)
        res = self.conv1(res)
        res = self.gn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.gn2(res)
        res = self.relu(res)
        if self.in_channels != self.out_channels:
            pad = [0] * (2 * len(identity.size()))
            pad[6] = (self.out_channels - self.in_channels)
            identity = F.pad(input=identity, pad=pad, mode='constant', value=0)
        return res + identity


class UNet3D(nn.Module):
    def __init__(self,
                 in_channel,
                 n_classes,
                 use_bias=True,
                 inplanes=32,
                 num_groups=8):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.inplanes = inplanes
        self.num_groups = num_groups
        planes = [inplanes * int(pow(2, i)) for i in range(0, 5)]
        super(UNet3D, self).__init__()
        self.ec0 = Encoder(in_channel,
                           planes[1],
                           bias=use_bias,
                           num_groups=num_groups)
        self.ec1 = Encoder(planes[1],
                           planes[2],
                           bias=use_bias,
                           num_groups=num_groups)
        self.ec1_2 = Encoder(planes[2],
                             planes[2],
                             bias=use_bias,
                             num_groups=num_groups)
        self.ec2 = Encoder(planes[2],
                           planes[3],
                           bias=use_bias,
                           num_groups=num_groups)
        self.ec2_2 = Encoder(planes[3],
                             planes[3],
                             bias=use_bias,
                             num_groups=num_groups)
        self.ec3 = Encoder(planes[3],
                           planes[4],
                           bias=use_bias,
                           num_groups=num_groups)
        self.ec3_2 = Encoder(planes[4],
                             planes[4],
                             bias=use_bias,
                             num_groups=num_groups)
        self.maxpool = nn.MaxPool3d(2)
        self.dc3 = Encoder(planes[4],
                           planes[4],
                           bias=use_bias,
                           num_groups=num_groups)
        self.dc3_2 = Encoder(planes[4],
                             planes[4],
                             bias=use_bias,
                             num_groups=num_groups)
        self.up3 = self.decoder(planes[4],
                                planes[3],
                                kernel_size=2,
                                stride=2,
                                bias=use_bias)
        self.dc2 = Encoder(planes[4],
                           planes[3],
                           bias=use_bias,
                           num_groups=num_groups)
        self.dc2_2 = Encoder(planes[3],
                             planes[3],
                             bias=use_bias,
                             num_groups=num_groups)
        self.up2 = self.decoder(planes[3],
                                planes[2],
                                kernel_size=2,
                                stride=2,
                                bias=use_bias)
        self.dc1 = Encoder(planes[3],
                           planes[2],
                           bias=use_bias,
                           num_groups=num_groups)
        self.dc1_2 = Encoder(planes[2],
                             planes[2],
                             bias=use_bias,
                             num_groups=num_groups)
        self.up1 = self.decoder(planes[2],
                                planes[1],
                                kernel_size=2,
                                stride=2,
                                bias=use_bias)
        self.dc0a = Encoder(planes[2],
                            planes[1],
                            bias=use_bias,
                            num_groups=num_groups)
        self.dc0b = self.decoder(planes[1],
                                 n_classes,
                                 kernel_size=1,
                                 stride=1,
                                 bias=use_bias,
                                 relu=False)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def decoder(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                output_padding=0,
                bias=True,
                relu=True):
        layer = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding,
                               bias=bias),
        ]
        if relu:
            layer.append(nn.GroupNorm(self.num_groups, out_channels))
            layer.append(nn.ReLU())
        layer = nn.Sequential(*layer)
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        e1 = self.ec1_2(self.ec1(self.maxpool(e0)))
        e2 = self.ec2_2(self.ec2(self.maxpool(e1)))
        e3 = self.ec3_2(self.ec3(self.maxpool(e2)))
        d3 = self.up3(self.dc3_2(self.dc3(e3)))
        if d3.size()[2:] != e2.size()[2:]:
            d3 = F.interpolate(d3,
                               e2.size()[2:],
                               mode='trilinear',
                               align_corners=False)
        d3 = torch.cat((d3, e2), 1)
        d2 = self.up2(self.dc2_2(self.dc2(d3)))
        if d2.size()[2:] != e1.size()[2:]:
            d2 = F.interpolate(d2,
                               e1.size()[2:],
                               mode='trilinear',
                               align_corners=False)
        d2 = torch.cat((d2, e1), 1)
        d1 = self.up1(self.dc1_2(self.dc1(d2)))
        if d1.size()[2:] != e0.size()[2:]:
            d1 = F.interpolate(d1,
                               e0.size()[2:],
                               mode='trilinear',
                               align_corners=False)
        d1 = torch.cat((d1, e0), 1)
        d0 = self.dc0b(self.dc0a(d1))
        return d0
