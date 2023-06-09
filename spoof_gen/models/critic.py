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
        
        self.aux_cls = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),)  #[16,16]
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),         #[128,128,128]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        #[256,64,64]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),)        #[512,32,32]
        self.critic = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32*32,1),
            nn.Sigmoid()
        )


    def forward(self,x, condition= None):
        # map_x = self.con_gen(x,condition)
        
        # map_x = self.upsampler(condition.unsqueeze(1))
        # print(map_x.shape)
        # print(x.shape)
        # input = torch.cat([x,map_x],dim = 1)

        input = x
        feature = self.feature_extractor(input)
        score = self.critic(feature)
        cls = self.aux_cls(feature)

        return score, cls
    
    def get_grad(self):
        grad = {}
        for name, param in self.named_parameters():
            grad[name] = (param.grad.clone().detach() if param.grad is not None else None)
        self.grad_ = grad
        del grad
    
    def load_grad(self):
        if hasattr(self,'grad_'):
            for name,param in self.named_parameters():
                if param.grad is not None:
                    param.grad.copy_(self.grad_[name])  
                else:
                    continue

            del self.grad_
