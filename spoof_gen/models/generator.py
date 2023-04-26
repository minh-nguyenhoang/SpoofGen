import torch
import torch.nn as nn
from .backbone import mobilenet, resnet, upblock, revnet



class Encoder(nn.Module):
    def __init__(self, model = 'mobilenet',) -> None:
        super().__init__()

        assert model in ['resnet-18','mobilenet','resnet-34','resnet-50','resnet-101','resnet-152']

        if 'resnet' in model:
            model, n_layers = model.split('-')
            self.backbone = resnet.resnet(n_layers)
        else:
            n_layers = None
            self.backbone = mobilenet.mobilenet()  #[128,16,16]

        self.encode = nn.Sequential(nn.Flatten(1),
                                    nn.Linear(128*16*16,32*32*2))
        
    def forward(self,x):
        z = self.backbone(x)
        z = self.encode(z)
        
        mu, l_var = torch.split(z,split_size_or_sections= 32*32, dim = 1)

        return mu, l_var


class Decoder(nn.Module):
    def __init__(self, model = 'unet',*, skip_z = True, bilinear = False, cdc = True) -> None:

        '''
        Other settings aside from model only affect the Unet-like Decoder, the reversed ResNet might be supported later. 
        '''

        super().__init__()
        assert model in ['unet','revnet-18','revnet-34','revnet-50','revnet-101']
        if model == 'unet':
            self.up = upblock.build_decoder_unet(skip_z= skip_z, bilinear= bilinear, cdc= cdc)
        else:
            model, n_layers = model.split('-')
            self.up = revnet.revnet(n_layers)

    def forward(self,z, condition):
        assert condition.shape[1] == 2

        inp = torch.cat([z,condition[:,0].unsqueeze(1),z,condition[:,1].unsqueeze(1)],dim = 1) #[N,4,32,32]
        return self.up(inp)
    
class Generator(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, con_gen) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.con_gen = con_gen

    def forward(self, x):
        r_condition = self.con_gen(x,1)
        s_condition = self.con_gen(x,0).cuda()
        condition = torch.stack([r_condition,s_condition], dim = 1)
        mu, log_sigma = self.encoder(x)

        # print(mu.shape, log_sigma.shape)

        z = mu + torch.randn((log_sigma.shape)).cuda() * torch.exp(log_sigma)
        z = z.view(z.shape[0],1,32,32)

        out = self.decoder(z,condition)

        return out


