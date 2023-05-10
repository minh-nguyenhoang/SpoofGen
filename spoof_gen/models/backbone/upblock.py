from ...utils import Conv2d_cd
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, basic_conv = Conv2d_cd):
        super().__init__()

        assert basic_conv in [Conv2d_cd, nn.Conv2d]
        if not mid_channels:
            mid_channels = int(out_channels)
        if isinstance(basic_conv, nn.Conv2d):
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                Conv2d_cd(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                Conv2d_cd(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Double conv then upscaling"""

    def __init__(self, in_channels, out_channels, bilinear=True, cdc = True):
        super().__init__()

        if cdc:
            conv = Conv2d_cd
        else:
            conv = nn.Conv2d

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True,)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, basic_conv= conv)
        else:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, basic_conv= conv)

    def forward(self, x):
        x = self.conv(x)
        return self.up(x)
    

class SkipUp(nn.Module):
    """Conv then upscaling"""

    def __init__(self, in_channels, out_channels, scale_factor, bilinear=True, cdc = True):
        super().__init__()

        if cdc:
            conv = Conv2d_cd
        else:
            conv = nn.Conv2d

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True,)
            self.conv = conv(in_channels,out_channels)
        else:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)
            self.conv = conv(in_channels,out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.up(x)
    
class UNetDecoder(nn.Module):
        def __init__(self, skip_z = True, bilinear = False, cdc = True) -> None:
            super().__init__()

            self.skip_z = skip_z
            self.bilinear = bilinear
            self.cdc = cdc

            ## 32 -> 32 -> 64 -> 128 -> 256
            
            self.first_conv = DoubleConv(2,64,basic_conv= Conv2d_cd)   # [N,64,32,32]
            self.l1 = Up(64,32,bilinear= self.bilinear, cdc= self.cdc)               # [N,32,64,64]
            self.l2 = Up(32,16,bilinear= self.bilinear, cdc= self.cdc)                # [N,16,128,128]
            self.l3 = Up(16,3,bilinear= self.bilinear, cdc= self.cdc)                # [N,3,256,256]

            if skip_z:
                self.skip_l1 = SkipUp(4,64,1,bilinear= self.bilinear, cdc= self.cdc)
                self.skip_l2 = SkipUp(4,32,2,bilinear= self.bilinear, cdc= self.cdc)
                self.skip_l3 = SkipUp(4,16,4,bilinear= self.bilinear, cdc= self.cdc)

        def forward(self,z):

            out = self.first_conv(z)

            if self.skip_z:
                out += self.skip_l1(z)

            out = self.l1(out)
            if self.skip_z:
                out += self.skip_l2(z)

            out = self.l2(out)
            if self.skip_z:
                out += self.skip_l3(z)

            out = self.l3(out)

            # return torch.stack(torch.split(out,split_size_or_sections= 3, dim = 1), dim = 0)
            return torch.sigmoid(out)   # [N,3,256,256]

def build_decoder_unet(skip_z = True, bilinear = False, cdc = True):
    
    return UNetDecoder(skip_z= skip_z, bilinear= bilinear, cdc= cdc)

            
