import torch 
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val : float, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ExponentialMeter(object):
    """Computes and stores the exponential average and current value"""
    def __init__(self, init : float= None, weight : float = .4):
        self.reset(init= init,weight = weight)

    def reset(self,init , weight : float = .4):
        assert 0 < weight <1
        self.weight = weight
        if init is not None:
            self.val = init
            self.avg = init
            self.count = 0
        else:          
            self.val = 0
            self.avg = 0
            self.count = 0

    def update(self, val : float):
        self.val = val
        self.count += 1
        self.avg = self.weight * val + (1 - self.weight) * self.avg
   
def get_infinite_data(data_loader, n = -1, init_step :int = None):
    step = 0 if init_step is None else init_step
    while True:
        for batch in data_loader:
            step +=1
            if step >= n:
                return
            yield step , batch



def contrast_depth_conv(input : torch.Tensor):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    device = input.device

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0],
                                             [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]
         ], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1,
                                                         0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(
        kernel_filter.astype(np.float)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.unsqueeze(dim=1).expand(
        input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(
        input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth

# Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
class Contrast_depth_loss:
    def __init__(self):
        pass

    def __call__(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''

        contrast_out = contrast_depth_conv(out)
        
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss()

        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)

        return loss
    

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class ConvTranpose2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, output_padding = 0, dilation=1, groups=1, bias=False, theta=0.7) -> None:
        super(ConvTranpose2d_cd, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
        mid_kernel = self.conv.weight[(kernel_size-1)//2][(kernel_size-1)//2]
        mid_kernel = mid_kernel[:,:,None,None]
        mid_kernel = torch.mul(mid_kernel, torch.ones(self.conv.weight.shape))

        self.conv.weight -= self.theta * mid_kernel

        out = self.conv(x)

        return out




def GlobalPooling(x: torch.Tensor):
    _, c, h, w = x.shape
    # return nn.Conv2d(c,c,w, groups= c)(x)
    # %timeit x.sum(2).sum(2).unsqueeze(-1).unsqueeze(-1)/(w*h)
    # %timeit nn.AvgPool2d(kernel_size = (w,h),stride = 1)(x)   
    # %timeit nn.AdaptiveAvgPool2d((1,1))(x)
    return nn.AdaptiveAvgPool2d((1,1))(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(SEBlock, self).__init__()
    
        self.global_pooling = GlobalPooling
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size= 1 , bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size= 1, bias = False)
        )
        self.sigmoid = torch.sigmoid


    def forward(self, x):

        out = self.global_pooling(x)
#         print(out.shape)
        out = self.se(out)
        
        out = self.sigmoid(out)
        
        
        return torch.mul(x,out)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

        
class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)

class SABlock(nn.Module):
    def __init__(self, kernel = 3) -> None:
        super().__init__()

        self.sa = SpatialAttention(kernel= kernel)

    def forward(self, x):
        sa_map = self.sa(x)
        return torch.mul(x,sa_map)
    


