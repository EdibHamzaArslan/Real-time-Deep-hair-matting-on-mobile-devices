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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)



class Encoder(nn.Module):
    def __init__(self, in_channels=32, out_channels_list=[64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 64
            DepthWiseEncoder(in_channels, out_channels_list[0], stride=1, padding=1),
            # 128
            DepthWiseEncoder(out_channels_list[0], out_channels_list[1], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[1], out_channels_list[1], stride=1, padding=1),
            # 256
            DepthWiseEncoder(out_channels_list[1], out_channels_list[2], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[2], out_channels_list[2], stride=1, padding=1),
            # 512
            DepthWiseEncoder(out_channels_list[2], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            # 1024
            DepthWiseEncoder(out_channels_list[3], out_channels_list[4], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[4], out_channels_list[4], stride=1, padding=1),
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels_list=[64, 64, 64]):
        super(Decoder, self).__init__()
        self.decoder_layer1 = DepthWiseDecoder(in_channels, out_channels_list[0], stride=1, padding=1)
        self.decoder_layer2 = DepthWiseDecoder(out_channels_list[0], out_channels_list[1], stride=1, padding=1)
        self.decoder_layer3 = DepthWiseDecoder(out_channels_list[1], out_channels_list[2], stride=1, padding=1)
    
    def forward(self, x):
        out1 = self.decoder_layer1(F.interpolate(x, scale_factor=2, mode="nearest"))
        out2 = self.decoder_layer2(F.interpolate(out1, scale_factor=2, mode="nearest"))
        out3 = self.decoder_layer3(F.interpolate(out2, scale_factor=2, mode="nearest"))
        return out3



class HairSegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(HairSegNet, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.encoder = Encoder(in_channels=32, out_channels_list=[64, 128, 256, 512, 1024])
        self.decoder = Decoder(in_channels=1024, out_channels_list=[64, 64, 64])
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.final_conv(x)



if __name__ == "__main__":

    x = torch.randn(1, 3, 224, 224)
    model = HairSegNet(3, 2)
    out = model(x)
    print(out.shape)
    assert out.shape == torch.Size([1, 2, 224, 224]), "Test Failed!"
    print("Test Passed!")
