import torch
import torch.nn as nn
from ..utils import SEBlock, SABlock
from .backbone.mobilenet import mobilenet
import warnings

class ConditionGenerator(nn.Module):
    def __init__(self, init_from = None) -> None:
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
        
        if init_from is not None:
            try:
                failed_states = self.load_pretrain_backbone(init_from)
            finally:
                if len(failed_states) != 0:
                    warnings.warn('During loading the weights, these states fail to load: '+ ', '.join(fs for fs in failed_states))
                else:
                    print('All states are loaded successfully!')
        

    def forward(self, x, condition):
        # for each batch of real images x, generate 1 batch of condition with 1 type (real/spoof)
        # So to generate 2 condition for the decoder and discriminator will take 2 forward pass
        N,_,_,_ = x.shape
        
        if isinstance(condition, int):
            condition = [condition]*N
        device = next(self.generator.parameters()).device
        output = []
        for cond in condition:
            if cond == 0:
                output.append((torch.rand((1,32,32))/10).abs().clip(0,1).to(device))
            else:
                output.append(self.last_conv(self.generator(x[len(output)].unsqueeze(0))).squeeze(1) + (torch.rand((1,32,32))/10).abs().clip(0,1).to(device))    #[N,32,32]
        
        return torch.cat(output,dim= 0)
    
    def load_pretrain_backbone(self, state_dict):

        own_state = self.state_dict()
        failed_states = []

        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, lambda storage, loc: storage)
            if 'model_checkpoint' in state_dict.keys():
                state_dict = state_dict['model_checkpoint']
        
        for name, param in state_dict.items():
            if name not in own_state:
                failed_states.append(name)
                # print(name)
                continue
            if own_state[name].shape != param.shape:
                failed_states.append(name)
                continue       
            own_state[name].copy_(param)
            # own_state[name].eval()
            # own_state[name].require_grads = False
            self._modules[name.split('.')[0]].eval()
            self._modules[name.split('.')[0]].require_grads = False
        
        return failed_states

