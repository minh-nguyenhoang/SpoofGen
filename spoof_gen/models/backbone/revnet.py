# ResNet with Convolution layer with stride != 1 become ConvTranspose for upscaling
# The ConvTranspose ops have kernel_size be the same as stride.
# The kernel size can be change to kernel_size = 2*padding + stride for upscaling, with dilation = 1 and output_padding = 0.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """
    PixelNorm PixelNorm from PG GAN
    thanks https://github.com/facebookresearch/pytorch_GAN_zoo/blob/b75dee40918caabb4fe7ec561522717bf096a8cb/models/networks/custom_layers.py#L9

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out


class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(RevBasicBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=stride+2, stride=stride, padding=1, bias=False)
        self.bn1 = PixelNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = PixelNorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=stride, stride=stride, bias=False),
                PixelNorm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RevBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(RevBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = PixelNorm(planes)
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=stride+2, stride=stride, padding=1, bias=False)
        self.bn2 = PixelNorm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = PixelNorm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=stride, stride=stride, bias=False),
                PixelNorm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RevResNet(nn.Module):
    def __init__(self, block, num_blocks, self_attn = True):
        super(RevResNet, self).__init__()
        self.in_planes = 16
        self.self_attn = self_attn
        self.conv1 = nn.Sequential(nn.Conv2d(2, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                                   PixelNorm(self.in_planes),
                                   nn.ReLU(inplace=True),
                                #    nn.Upsample(scale_factor=2, mode= 'bilinear')
                                   )
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 24, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8, num_blocks[3], stride=2)

        # last layer to compress channel down to 3
        self.last_conv = nn.Sequential(nn.Conv2d(8 * block.expansion,3,kernel_size = 1),
                                       nn.Sigmoid())

        
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        if self.self_attn:
            layers.append(Self_Attn(self.in_planes,'relu'))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x [N,4,16,16]
        out = self.conv1(x)                #[N,16,16,16]
        out = self.layer1(out)             #[N,32,32,32]
        out = self.layer2(out)             #[N,24,64,64] 
        out = self.layer3(out)             #[N,16,128,128]
        out = self.layer4(out)             #[N,8,256,256]
        
        out = self.last_conv(out)          #[N,3,256,256]
        # return torch.stack(torch.split(out,split_size_or_sections= 3, dim = 1), dim = 0)  # [2,N,3,256,256]
        return out
    
def revnet(mode = '18', self_attn= True):
    assert mode in ['18','34','50','101','152']
    if mode == '18':
        return RevResNet(RevBasicBlock, [2, 2, 2, 2], self_attn)
    if mode == '34':
        return RevResNet(RevBasicBlock, [3, 6, 4, 3], self_attn)
    if mode == '50':
        return RevResNet(RevBottleneck, [3, 6, 4, 3], self_attn)
    if mode == '101':
        return RevResNet(RevBottleneck, [3, 23, 4, 3], self_attn)
    if mode == '152':
        return RevResNet(RevBottleneck, [3, 36, 8, 3], self_attn)

