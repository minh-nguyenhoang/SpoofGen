import torch
import torch.nn as nn
import torch.nn.functional as F
# from conditional_generator import ConditionGenerator
from ..utils import Conv2d_cd


class ScaledTanh(nn.Module):
    def __init__(self, scale = 10) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad= False)
    
    def forward(self, x):
        return self.scale * F.tanh(x)

class Critic(nn.Module):
    def __init__(self, input_dims = (256,256,3), score_scale = 10) -> None:
        super().__init__()
        self.input_dims = input_dims

        # self.con_gen = ConditionGenerator()
        # self.con_gen.eval()
        # for params in self.con_gen.parameters():
        #     params.requires_grad = False

        self.upsampler = nn.Upsample((input_dims[0],input_dims[1]),mode = 'bilinear')

        self.critic = nn.Sequential(
            Conv2d_cd(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            Conv2d_cd(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),         #[128,128,128]
            Conv2d_cd(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #[256,64,64]
            Conv2d_cd(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #[512,32,32]
            Conv2d_cd(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            Conv2d_cd(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32*32,1),
            ScaledTanh(score_scale)
        )


    def forward(self,x, condition):
        # map_x = self.con_gen(x,condition)
        
        map_x = self.upsampler(condition.unsqueeze(1))
        # print(map_x.shape)
        # print(x.shape)
        input = torch.cat([x,map_x],dim = 1)

        score = self.critic(input)

        return score
    
    def get_grad(self):
        grad = []
        for param in self.parameters():
            grad.append(param.grad.clone().detach())
        self.grad_ = grad
        del grad
    
    def load_grad(self):
        for param in self.parameters():
            param.grad.copy_(self.grad_.pop(0))

        del self.grad_