import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DepthWiseEncoder, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DepthWiseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DepthWiseDecoder, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)



class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels_list=[32, 64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        # 32
        self.conv1 = nn.Conv2d(in_channels, out_channels_list[0], kernel_size=3, stride=2, padding=1)
        self.encoder = nn.Sequential(
            # 64
            DepthWiseEncoder(out_channels_list[0], out_channels_list[1], stride=1, padding=1),
            # 128
            DepthWiseEncoder(out_channels_list[1], out_channels_list[2], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[2], out_channels_list[2], stride=1, padding=1),
            # 256
            DepthWiseEncoder(out_channels_list[2], out_channels_list[3], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            # 512
            DepthWiseEncoder(out_channels_list[3], out_channels_list[4], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[4], out_channels_list[4], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[4], out_channels_list[4], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[4], out_channels_list[4], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[4], out_channels_list[4], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[4], out_channels_list[4], stride=1, padding=1),
            # 1024
            DepthWiseEncoder(out_channels_list[4], out_channels_list[5], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[5], out_channels_list[5], stride=1, padding=1),
        )
    
    def forward(self, x):
        return self.encoder(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels_list=[64, 64, 64]):
        super(Decoder, self).__init__()
        self.decoder_layer1 = DepthWiseDecoder(in_channels, out_channels_list[0], stride=1, padding=1)
        self.deceder_layer2 = DepthWiseDecoder(out_channels_list[0], out_channels_list[1], stride=1, padding=1)
        self.decoder_layer3 = DepthWiseDecoder(out_channels_list[1], out_channels_list[2], stride=1, padding=1)
    
    def forward(self, x):
        out1 = self.decoder_layer1(F.interpolate(x, scale_factor=2, mode="nearest"))
        out2 = self.decoder_layer2(F.interpolate(out1, scale_factor=2, mode="nearest"))
        out3 = self.decoder_layer3(F.interpolate(out2, scale_factor=2, mode="nearest"))



class HairSegNet(nn.Module):
    def __init__(self):
        super(HairSegNet, self).__init__()
        




class HairMatteNet(nn.Module):
    pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
