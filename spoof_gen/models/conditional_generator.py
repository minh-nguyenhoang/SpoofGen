import torch
import torch.nn as nn
from ..utils import SEBlock, SABlock
from .backbone.mobilenet import mobilenet

class ConditionGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.generator = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     SEBlock(32,4),
        #     SABlock(),   
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),         #[128,128,128]
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     SEBlock(64,4),
        #     SABlock(), 
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #[256,64,64]
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     SEBlock(64,4),
        #     SABlock(),   
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #[512,32,32]
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        # )

        self.generator = mobilenet()
        self.last_conv = nn.Sequential(nn.ConvTranspose2d(128,1,2,2), 
                                       nn.ReLU(inplace= True))
        

    def forward(self, x, condition):
        # for each batch of real images x, generate 1 batch of condition with 1 type (real/spoof)
        # So to generate 2 condition for the decoder and discriminator will take 2 forward pass
        N,_,_,_ = x.shape
        if condition == 0:
            return torch.rand((N,32,32)).abs().clip(0,0.15)
        else:
            return self.last_conv(self.generator(x)).squeeze(1)                 #[N,32,32]
