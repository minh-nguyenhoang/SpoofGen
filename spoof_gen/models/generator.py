import torch
import torch.nn as nn
from .backbone import mobilenet, resnet, upblock, revnet, iresnet
import warnings

ENCODE_SIZE = 16*16
SQRT_ENCODE_SIZE = torch.sqrt(torch.tensor(ENCODE_SIZE)).int().item()


class Encoder(nn.Module):
    def __init__(self, model = 'iresnet-50', init_from = None) -> None:
        super().__init__()

        assert model in ['iresnet-18','iresnet-34','iresnet-50','iresnet-100','iresnet-200']

        if 'iresnet' in model:
            model, n_layers = model.split('-')
            self.backbone = iresnet.iresnet(n_layers,ENCODE_SIZE)
        else:
            raise Exception("Please choose a valid IResNet model, the mobilenet version is not supported right now!")
        # else:
        #     n_layers = None
        #     self.backbone = mobilenet.mobilenet()  #[128,8,8]

        # self.encode = nn.Sequential(nn.Flatten(1),
        #                             nn.Linear(512*expansion*16*16,ENCODE_SIZE*2))

        if init_from is not None:
            failed_states = []
            try:
                failed_states = self.load_pretrain_backbone(init_from)
            finally:
                if len(failed_states) != 0:
                    warnings.warn('During loading the weights, these states fail to load: '+ ', '.join(fs for fs in failed_states))
                else:
                    print('All states are loaded successfully!')
        
    def forward(self,x):
        z = self.backbone(x)
        # z = self.encode(z)
        
        mu, l_var = torch.split(z,split_size_or_sections= ENCODE_SIZE, dim = 1)

        return mu, l_var
    
    def load_backbone_state_dict(self, state_dict, strict: bool = True):
        return self.backbone.load_state_dict(state_dict, strict)

    def load_pretrain_backbone(self, state_dict):

        own_state = self.backbone.state_dict()
        failed_states = []
        failed_layers=[]

        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, lambda storage, loc: storage)
        
        for name, param in state_dict.items():
            
            if name not in own_state:
                failed_states.append(name)
                failed_layers.append(name.split('.')[0])
                # print(name)
                continue
            if own_state[name].shape != param.shape:
                failed_states.append(name)
                failed_layers.append(name.split('.')[0])
                self.backbone._modules[name.split('.')[0]].train()
                continue       
            else:
                if name.split('.')[0] not in failed_layers:
                    own_state[name].copy_(param)
                    if not isinstance((module := self.backbone._modules[name.split('.')[0]]), torch.nn.BatchNorm1d):
                        module.eval()
        
        return failed_states




class Decoder(nn.Module):
    def __init__(self, model = 'unet',*, skip_z = True, bilinear = False, cdc = True, self_attn= False) -> None:

        '''
        Parameters:
            - skip_z: Wether to append the latent embbeding to each layer of the model. Default: True.\n
            - bilinear: Wether to use bilinear upsampling for the model, else use TransposedConvolution. Default: False.\n
            - cdc: Wether to use central difference convolution for the Convolution operator. Default: True.\n
            - self_attn: Wether to use self attention as describe in Self-Attention GAN. Default: False.\n
        Other settings aside from model and self_attn only affect the Unet-like Decoder, the reversed ResNet might be supported later. 
        '''

        super().__init__()
        assert model in ['unet','revnet-18','revnet-34','revnet-50','revnet-101']
        if model == 'unet':
            self.up = upblock.build_decoder_unet(skip_z= skip_z, bilinear= bilinear, cdc= cdc)
        else:
            model, n_layers = model.split('-')
            self.up = revnet.revnet(n_layers, self_attn)

    def forward(self,z, condition):

        inp = torch.cat([z,condition.unsqueeze(1)],dim = 1) #[N,2,32,32]
        # return torch.sigmoid(self.up(inp))
        return self.up(inp)
    
class Generator(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, con_gen: nn.Module) -> None:
        super().__init__()
        '''
        For evaluation
        '''
        self.encoder = encoder
        self.decoder = decoder
        self.con_gen = con_gen

        # self.encoder.eval()
        # for params in self.encoder.parameters():
        #     params.requires_grad = False


        self.con_gen.eval()
        for params in self.con_gen.parameters():
            params.requires_grad = False

    def forward(self, x, condition):
        # assert condition in [0,1], "The condition must be 0 or 1!"
        device = next(self.parameters()).device
        condition = self.con_gen(x,condition).clone().detach()
        # condition = torch.stack([r_condition,s_condition], dim = 1)
        mu, log_var = self.encoder(x)

        # print(mu.shape, log_sigma.shape)

        z = mu + torch.rand_like(log_var).to(device) \
                            * torch.exp(0.5*log_var)
        z = z.view(z.shape[0],1,SQRT_ENCODE_SIZE,SQRT_ENCODE_SIZE)

        out = self.decoder(z,condition)    # [N,3,256,256]

        return out, condition, mu, log_var
    
    def sample(self,
            num_samples:int,
            label,
            **kwargs) -> torch.Tensor:
        '''
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        '''

        current_device = next(self.decoder.parameters()).device
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        32*32)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decoder(z)
        return samples

    def generate(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, *args, **kwargs)[0]


