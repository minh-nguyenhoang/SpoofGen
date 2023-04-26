import torch
import torch.nn as nn
from conditional_generator import ConditionGenerator
from utils import Conv2d_cd


class Critic(nn.Module):
    def __init__(self, input_dims = (256,256,3), out_dims = 2) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.out_dims = out_dims

        self.con_gen = ConditionGenerator()
        self.con_gen.eval()
        for params in self.con_gen.parameters():
            params.requires_grad = False

        self.upsampler = nn.Upsample((input_dims[0],input_dims[1]),mode = 'bilinear')

        self.critic = nn.Sequential(
            Conv2d_cd(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Conv2d_cd(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),         #[128,128,128]
            Conv2d_cd(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #[256,64,64]
            Conv2d_cd(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #[512,32,32]
            Conv2d_cd(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*32,1)
        )


    def forward(self,x, condition):
        # map_x = self.con_gen(x,condition)
        map_x = self.upsampler(condition).unsqueeze(1)
        input = torch.stack([x,map_x],dim = 1)

        score = self.critic(input)

        return score