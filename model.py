import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DepthWiseEncoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)


class DepthWiseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DepthWiseDecoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv_block(x)



class Encoder(nn.Module):
    def __init__(self, in_channels=32, out_channels_list=[64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        self.encoder_layer1 = nn.Sequential(
            # 64
            DepthWiseEncoder(in_channels, out_channels_list[0], stride=1, padding=1),
        )
        self.encoder_layer2 = nn.Sequential(
            # 128
            DepthWiseEncoder(out_channels_list[0], out_channels_list[1], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[1], out_channels_list[1], stride=1, padding=1),
        )
        self.encoder_layer3 = nn.Sequential(
            # 256
            DepthWiseEncoder(out_channels_list[1], out_channels_list[2], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[2], out_channels_list[2], stride=1, padding=1),
        )
        self.encoder_layer4 = nn.Sequential(
            # 512
            DepthWiseEncoder(out_channels_list[2], out_channels_list[3], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1),
            DepthWiseEncoder(out_channels_list[3], out_channels_list[3], stride=1, padding=1)
            
        )
        self.encoder_layer5 = nn.Sequential(
            # 1024
            DepthWiseEncoder(out_channels_list[3], out_channels_list[4], stride=2, padding=1),
            DepthWiseEncoder(out_channels_list[4], out_channels_list[4], stride=1, padding=1),
        )
    
    def forward(self, x):
        x_res1 = self.encoder_layer1(x)
        x_res2 = self.encoder_layer2(x_res1)
        x_res3 = self.encoder_layer3(x_res2)
        x_res4 = self.encoder_layer4(x_res3)
        x = self.encoder_layer5(x_res4)
        return x, x_res1, x_res2, x_res3, x_res4


class Decoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels_list=[64, 64, 64, 64, 64]):
        super(Decoder, self).__init__()
        self.decoder_block1 = DepthWiseDecoder(in_channels, out_channels_list[0], stride=1, padding=1)
        self.decoder_block2 = DepthWiseDecoder(out_channels_list[0], out_channels_list[1], stride=1, padding=1)
        self.decoder_block3 = DepthWiseDecoder(out_channels_list[1], out_channels_list[2], stride=1, padding=1)
        self.decoder_block4 = DepthWiseDecoder(out_channels_list[2], out_channels_list[3], stride=1, padding=1)
        self.decoder_block5 = DepthWiseDecoder(out_channels_list[3], out_channels_list[4], stride=1, padding=1)
    
    def forward(self, x, res1, res2, res3, res4):
        out1 = self.decoder_block1(res4 + F.interpolate(x, scale_factor=2, mode="nearest"))
        out2 = self.decoder_block2(res3 + F.interpolate(out1, scale_factor=2, mode="nearest"))
        out3 = self.decoder_block3(res2 + F.interpolate(out2, scale_factor=2, mode="nearest"))
        out4 = self.decoder_block4(res1 + F.interpolate(out3, scale_factor=2, mode="nearest"))
        out5 = self.decoder_block5(F.interpolate(out4, scale_factor=2, mode="nearest"))
        return out5



class HairMatteNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(HairMatteNet, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.encoder = Encoder(in_channels=32, out_channels_list=[64, 128, 256, 512, 1024])
        self.res_conv_layer1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.res_conv_layer2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.res_conv_layer3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.res_conv_layer4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.decoder = Decoder(in_channels=1024, out_channels_list=[64, 64, 64, 64, 64])
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.first_conv(x)
        x, encoder_res1_out, encoder_res2_out, encoder_res3_out, encoder_res4_out = self.encoder(x)
        
        res_out1 = self.res_conv_layer1(encoder_res1_out)
        res_out2 = self.res_conv_layer2(encoder_res2_out)
        res_out3 = self.res_conv_layer3(encoder_res3_out)
        res_out4 = self.res_conv_layer4(encoder_res4_out)

        x = self.decoder(x, res_out1, res_out2, res_out3, res_out4)
        return self.final_conv(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # import torchvision
    # model = torchvision.models.mobilenet_v2(pretrained=True)
    # print(model)

    # import timm
    # available_models = timm.list_models("*mobilenet*")
    # # for model_name in available_models:
    # #     print(model_name)
    # model = timm.create_model("mobilenetv2_035")
    # print(model)
    
    x = torch.randn(1, 3, 224, 224)
    model = HairMatteNet(3, 2)
    # print(model)
    # print(model.encoder.encoder_block[0].block[3])
    out = model(x)
    print(out.shape)
    assert out.shape == torch.Size([1, 2, 224, 224]), "Test Failed!"
    print("Test Passed!")
