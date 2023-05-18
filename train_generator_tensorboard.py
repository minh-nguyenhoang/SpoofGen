import comet_ml
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from spoof_gen.data_utils import StandardDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from spoof_gen.models.generator import Encoder, Decoder, Generator
from spoof_gen.models.conditional_generator import ConditionGenerator
from spoof_gen.models.critic import Critic
from spoof_gen.utils import AverageMeter, ExponentialMeter, get_infinite_data
## from tensorboardX import SummaryWriter
from spoof_gen.logging import SummaryWriter
import torch.optim as opt
import lightning as L
import os
from tqdm.auto import tqdm
from torchsummary import summary


#################################
######### AMP TRAINING ##########
#################################
USE_AMP = False
torch.backends.cudnn.benchmark = False
scaler = torch.cuda.amp.GradScaler(init_scale=4096.0)


fabric = L.Fabric(accelerator="cuda", devices=1, strategy="auto")
fabric.launch()
fabric.seed_everything(1273)

def create_logging_plot(inputs: torch.Tensor,condition: torch.Tensor, generated: torch.Tensor):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np
    norm = Normalize(0,1, True)
    plt.cla()  # clear previous figure to free memory used
    plt.clf()
    N,_,_,_ = inputs.shape

    fig,ax = plt.subplots(3,N, figsize= (20,12), squeeze= False)
    i = 0
    while i< N:
        idx = np.random.randint(0,N)
        ax[0,i].imshow(inputs[idx].permute(1,2,0).cpu().numpy());
        ax[0,i].axis('off');

        ax[1,i].imshow(condition[idx].cpu().numpy(), cmap= 'gray', norm= norm);
        ax[1,i].axis('off');

        ax[2,i].imshow(generated[idx].permute(1,2,0).detach().cpu().numpy());
        ax[2,i].axis('off');

        i += 1

    return fig


##################################
####### GENERAL SETTINGS #########
##################################
VALIDATION_CHECKER = 'no_grad'

CRITIC_CHECKPOINT = None
ENCODER_CHECKPOINT =  None
DECODER_CHECKPOINT = None

EPS = 1e-3 if USE_AMP else 1e-9

##################################
####### TRAINING SETTINGS ########
##################################
STEPS = 100000
BATCH_SIZE = 1
ACCUMULATED_OPTIMIZER_STEP = 4
VAL_EPOCH_EVERY_N_TRAIN_STEPS = 1000
SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS = 1

GENERATOR_STEP_EVERY_N_CRITIC_STEP = 5
GRADIENT_PENALTY_WEIGHT = 10

def gradient_penalty(critic_model, real_data, generated_data, condition, writer):
    batch_size = real_data.shape[0]

    device = next(crit_model.parameters()).device
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)

    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True)


    # Calculate probability of interpolated examples
    prob_interpolated = critic_model(interpolated, condition)

    # Calculate gradients of probabilities with respect to examples
    if USE_AMP:
        gradients = torch_grad(outputs=scaler.scale(prob_interpolated), inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.shape).to(device),
                            create_graph=True, retain_graph=True)[0]
    
        gradients = gradients / scaler.get_scale()
    else:
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.shape).to(device),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    if writer is not None:
        if critic_model.training:
            writer.add_scalar('loss/train_gradient_norm',gradients.norm(2, dim=1).mean().item())

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return GRADIENT_PENALTY_WEIGHT * ((gradients_norm - 1) ** 2).mean()

##################################
###### CHECKPOINT DIRECTORY ######
##################################
CKPT_DIR = "checkpoints"

if not CKPT_DIR.endswith('/'):
    CKPT_DIR += '/'

SUB_CKPT_DIR = ['critic', 'encoder', 'decoder']

for i in range(len(SUB_CKPT_DIR)):
    if not SUB_CKPT_DIR[i].endswith('/'):
        SUB_CKPT_DIR[i] += '/'

MODEL_NAME = {'critic/': 'critic','encoder/': 'encoder_arcface_18','decoder/':'decoder_revnet_18'}

SAVE_BEST = True

def save_checkpoint(sub_dir,*,
                    model, optimizer:torch.optim.Optimizer = None, scheduler:torch.optim.lr_scheduler._LRScheduler = None, 
                    loss_meter : AverageMeter|ExponentialMeter = None, best_loss = None, step:int = None, save_best = SAVE_BEST, writer = None):
    assert not save_best or (loss_meter is not None and best_loss is not None and step is not None), 'If you want to save the best checkpoint, please provide a loss meter, current best loss and step!'
    
    ## Just for Validation checker before actually running the training process
    if 'test' in sub_dir:
        sub_dir = 'encoder/'
        test = True
    else:
        test = False
    
    model.half()
    if save_best and loss_meter.avg < best_loss:
        for ckpt in os.listdir(f'{CKPT_DIR}{sub_dir}'):
            os.remove(f'{CKPT_DIR}{sub_dir}{ckpt}')
        best_loss = loss_meter.avg
        torch.save({'model_checkpoint': model.state_dict(),
                    'optimizer_checkpoint': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler_checkpoint': scheduler.state_dict() if scheduler is not None else None,
                    'step': step,
                    'loss': loss_meter.avg,
                    }, f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_best.pth')
        # if not test and writer is not None:
        #     writer[f'checkpoint/{sub_dir}best'].upload(f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_best.pth')
        
    torch.save({'model_checkpoint': model.state_dict(),
                'optimizer_checkpoint': optimizer.state_dict() if optimizer is not None else None,
                'scheduler_checkpoint': scheduler.state_dict() if scheduler is not None else None,
                'step': step if step is not None else None,
                'loss': loss_meter.avg if loss_meter is not None else None,
                }, f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_last.pth')
    # if not test and writer is not None:
    #     writer[f'checkpoint/{sub_dir}last'].upload(f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_last.pth')
    model.float()

    if test:
        os.remove(f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_last.pth')
        if save_best and abs(loss_meter.avg - best_loss) <1e-3:
            os.remove(f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_best.pth')
    
    return best_loss


##################################
###### OPTIMIZER & SCHDULER ######
##################################
LR = 1e-4
BETA_1 = .5
BETA_2 = .999
STEP_SIZE = STEPS // 3
GAMMA = 0.3



##################################
############## DATA ##############
##################################

print('Processing data')
transform = transforms.Compose([transforms.RandomRotation(15), transforms.RandomHorizontalFlip()])

train_data = StandardDataset('Data/train_img', transform= transform)
train_loader = DataLoader(train_data, batch_size= BATCH_SIZE, shuffle= True)

val_data = StandardDataset('Data/test_img', transform= None)
sub_val = Subset(val_data, range(40 * int(1.5*BATCH_SIZE)))
val_loader = DataLoader(sub_val, batch_size= int(1.5*BATCH_SIZE), shuffle= True)


##################################
## MODEL, LOSS, OPTIMIZER, ECT. ##
##################################

encoder = Encoder(model= 'iresnet-18', init_from='checkpoints/misc/backbone_arcface_18.pth')
decoder = Decoder(model= 'revnet-18')
con_gen = ConditionGenerator(output_shape= 16, init_from='checkpoints\conditional_generator/conditonal_generator_mobilenetv3_last.pth')

gen_model = Generator(encoder= encoder, decoder= decoder, con_gen= con_gen)

gen_optimizer = opt.Adam(gen_model.parameters(),lr = LR, betas= (BETA_1, BETA_2))
gen_scheduler = opt.lr_scheduler.StepLR(optimizer= gen_optimizer, step_size= STEP_SIZE, gamma= GAMMA)


crit_model = Critic()

crit_optimizer = opt.Adam(crit_model.parameters(),lr = LR, betas= (BETA_1, BETA_2))
crit_scheduler = opt.lr_scheduler.StepLR(optimizer= crit_optimizer, step_size= STEP_SIZE, gamma= GAMMA)



loss_mse = nn.MSELoss()


##################################
##### SETUP LIGHTNING FABRIC #####
##################################
gen_model, gen_optimizer, gen_scheduler = fabric.setup(gen_model, gen_optimizer, gen_scheduler)
crit_model, crit_optimizer, crit_scheduler = fabric.setup(crit_model, crit_optimizer, crit_scheduler)

train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)


## Load the previous training session weights if possible. 
## Bring this behind fabric setup follow advices on 
## https://discuss.pytorch.org/t/expected-all-tensors-to-be-on-the-same-device-but-found-at-least-two-devices-cuda-0-and-cpu/101693/7
## TODO: rewrite this to a function (DONE!)

def load_checkpoint(ckpt, model, optimizer= None, scheduler= None):
    ckpt = torch.load(ckpt, lambda storage, loc: storage)
    model.load_state_dict(ckpt['model_checkpoint'])
    if optimizer is not None and ckpt['optimizer_checkpoint'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_checkpoint'])
    if scheduler is not None and ckpt['scheduler_checkpoint'] is not None:
        scheduler.load_state_dict(ckpt['scheduler_checkpoint'])
    step = ckpt['step']
    loss = ckpt['loss']

    return step, loss

crit_step, encoder_step, decoder = 0
init_crit_loss, init_encoder_loss, init_decoder_loss = 1000
if CRITIC_CHECKPOINT is not None:
    crit_step, init_crit_loss = load_checkpoint(CRITIC_CHECKPOINT, crit_model, crit_optimizer, crit_scheduler)
if ENCODER_CHECKPOINT is not None:
    encoder_step, init_encoder_loss = load_checkpoint(ENCODER_CHECKPOINT, gen_model.encoder, gen_optimizer, gen_scheduler)
if DECODER_CHECKPOINT is not None:
    decoder_step, init_decoder_loss = load_checkpoint(DECODER_CHECKPOINT, gen_model.decoder, None, None)

init_step = max([crit_step, encoder_step, decoder_step])
init_gen_loss = min(init_decoder_loss, init_encoder_loss)

##################################
###### TENSORBOARDX LOGGER #######
##################################

writer = SummaryWriter(comet_config = {  
                "api_key" : "0I4vcVpuefcoLxavIPszjBsyF",
                "project_name" : "spoof-gen",
                "workspace": "minh-nguyenhoang",
                "disabled": False
})


hparams = {'steps': STEPS,'batch_size': BATCH_SIZE,'accumalated_steps': ACCUMULATED_OPTIMIZER_STEP, "optimizer": "Adam", "learning_rate": LR, 'scheduler': "StepLR",
           'step_size': STEP_SIZE, 'gamma': GAMMA}

writer.add_hparams(
    hparam_dict=hparams,
    metric_dict={}
)




##################################
#### TRAINING AND VALIDATING #####
##################################

## Validation checker
if VALIDATION_CHECKER == 'no_grad':
    test_meter= ExponentialMeter(weight= .3)
    BREAK_POINT = 5

    print("Start checking")
    check_loader = DataLoader(Subset(val_data, range(BREAK_POINT * int(1.5*BATCH_SIZE))), batch_size= int(1.5*BATCH_SIZE), shuffle= False)
    check_loader = fabric.setup_dataloaders(check_loader)
    try:
        with torch.no_grad():
            for idx, batch in enumerate(pbar := tqdm(check_loader,position= 0, desc= 'Validation checker')):
                inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
                with torch.cuda.amp.autocast(enabled= USE_AMP):
                    ## create a generated batch 
                    generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
                    
                    print(mu.mean().item(), log_var.mean().item())
                    ### critic take action
                    d_real = crit_model(inputs, condition)
                    d_gen = crit_model(generated.detach(), condition)
        
                    gen_mu, gen_log_var = gen_model.encoder(generated)
        
                    ### calculating the loss
        
                    crit_loss = d_gen.mean() - d_real.mean() + 2
                    
                    d_gen1 = crit_model(generated, condition)
                    rec_loss = loss_mse(inputs, generated)
                    kl_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                    sim_loss = torch.mean(-0.5 * torch.mean(log_var - gen_log_var - ((gen_mu - mu) ** 2)/(gen_log_var.exp() + EPS) - (log_var - gen_log_var).exp(), dim = 1), dim = 0)
                    gen_loss = (rec_loss.mean() + .2 * kl_loss + .2 * sim_loss) - d_gen1.mean() + 1

                # if idx in range(2,4):
                #     print('\nReconstruction loss: ', rec_loss.float().item())
                #     print('KL-Divergence loss: ', kl_loss.float().item())
                #     print('Similarity loss: ', sim_loss.float().item())
                #     print('Real loss', d_real.mean().float().item())
                #     print('Fake loss', -d_gen.mean().float().item())
                
                # if (idx+1) % 5 == 0:
                #     writer.add_figure('valiation_check_output',create_logging_plot(inputs, condition, generated.float())
                
                test_meter.update(gen_loss.float().item())

                # input('Press Enter!')

                pbar.set_postfix({'crit_loss': crit_loss.item(), 'gen_loss': gen_loss.item()})
                
            ## Initialize some default values for 
            gen_train_avm = ExponentialMeter(init= test_meter.avg, weight = .3)
                

            ## test save
        
            loss = save_checkpoint('test/', model= gen_model.encoder, optimizer= gen_optimizer, scheduler= gen_scheduler, \
                                        loss_meter= test_meter ,best_loss= 1000, epoch= -1, save_best= True)
            


    except Exception as e:
        print('>>>>' + repr(e))
        exit()


elif VALIDATION_CHECKER:
    test_meter= ExponentialMeter(weight= .3)
    BREAK_POINT = 5

    print("Start checking")
    check_loader = DataLoader(Subset(val_data, range(BREAK_POINT * int(1.5*BATCH_SIZE))), batch_size= int(1.5*BATCH_SIZE), shuffle= False)
    check_loader = fabric.setup_dataloaders(check_loader)
    try:
        for idx, batch in enumerate(pbar := tqdm(check_loader,position= 0, desc= 'Validation checker')):
            inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
            with torch.cuda.amp.autocast(enabled= USE_AMP):
                ## create a generated batch 
                generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
                
                print(mu.mean().item(), log_var.mean().item())
                ### critic take action
                d_real = crit_model(inputs, condition)
                d_gen = crit_model(generated.detach(), condition)
    
                gen_mu, gen_log_var = gen_model.encoder(generated)
                with torch.cuda.amp.autocast(enabled= False):
                    gp = gradient_penalty(critic_model= crit_model, real_data= inputs, generated_data= generated.detach(), condition= condition,\
                                        writer= None)
    
                ### calculating the loss
    
                crit_loss = d_gen.mean() - d_real.mean() + gp + 2
                
                d_gen1 = crit_model(generated, condition)
                rec_loss = loss_mse(inputs, generated)
                kl_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                sim_loss = torch.mean(-0.5 * torch.mean(log_var - gen_log_var - ((gen_mu - mu) ** 2)/(gen_log_var.exp() + EPS) - (log_var - gen_log_var).exp(), dim = 1), dim = 0)
                gen_loss = (rec_loss.mean() + .2 * kl_loss + .2 * sim_loss) - d_gen1.mean() + 1

            # if idx in range(2,4):
            #     print('\nReconstruction loss: ', rec_loss.float().item())
            #     print('KL-Divergence loss: ', kl_loss.float().item())
            #     print('Similarity loss: ', sim_loss.float().item())
            #     print('Real loss', d_real.mean().float().item())
            #     print('Fake loss', -d_gen.mean().float().item())


            if USE_AMP:
                fabric.backward(scaler.scale(crit_loss))
            else:
                fabric.backward(crit_loss)

            
            # if (idx+1) % 5 == 0:
            #     writer.add_figure('valiation_check_output',create_logging_plot(inputs, condition, generated.float()))
            
            crit_model.get_grad()
            crit_model.load_grad()
            crit_optimizer.zero_grad()
            # if USE_AMP:
            #     scaler.step(crit_optimizer)
            # else:
            #     crit_optimizer.step()
            
            

            if USE_AMP:
                fabric.backward(scaler.scale(gen_loss))
                gen_optimizer.zero_grad()
                # scaler.step(gen_optimizer)
                # scaler.update()
            else:
                fabric.backward(gen_loss)
                gen_optimizer.zero_grad()
                # gen_optimizer.step()
            
            test_meter.update(gen_loss.float().item())

            # input('Press Enter!')

            pbar.set_postfix({'crit_loss': crit_loss.item(), 'gen_loss': gen_loss.item()})
            
        ## Initialize some default values for 
        gen_train_avm = ExponentialMeter(init= test_meter.avg, weight = .3)
            

        ## test save
    
        loss = save_checkpoint('test/', model= gen_model.encoder, optimizer= gen_optimizer, scheduler= gen_scheduler, \
                                    loss_meter= test_meter ,best_loss= 1000, epoch= -1, save_best= True)
            


    except Exception as e:
        print('>>>>' + repr(e))
        exit()

    # finally:
    #     ## just for checking, remove this when training

    #     raw = input("Seem OK, press Enter to exit")
    #     exit()



raw = input("Seem OK, press Enter to continue or Q to exit.")
if raw.lower() == 'q':
    exit()

print("######################################")
print("#########  START TRAINING  ###########")
print("######################################")

# exit()

##################################
####### SETUP LOSS METER #########
##################################

## The use of exponential smoothing over averaging for easier weighting of current value

crit_best_loss = torch.inf
gen_best_loss = torch.inf

gen_val_avm = ExponentialMeter( weight = .3)
crit_train_avm = ExponentialMeter( weight = .3)
crit_val_avm = ExponentialMeter( weight = .3)


## Decompose gen_model.train() to several components, the remainings are not being updated.
gen_model.decoder.train()
gen_model.encoder.fc.train()
gen_model.encoder.features.train()
crit_model.train()

for step, batch in enumerate(pbar := tqdm(get_infinite_data(train_loader,STEPS, init_step= init_step), position=0, leave= False, desc='Training')):
    inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
    with torch.cuda.amp.autocast(enabled= USE_AMP):
        ## create a generated batch 
        generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)

    if step % (ACCUMULATED_OPTIMIZER_STEP * (GENERATOR_STEP_EVERY_N_CRITIC_STEP + 1)) < ACCUMULATED_OPTIMIZER_STEP * (GENERATOR_STEP_EVERY_N_CRITIC_STEP ):

        #####################
        ## CRITIC TRAINING ##
        #####################

        ### Detach the generated images so the grad of generator will not be update here
        ### critic take action
        with torch.cuda.amp.autocast(enabled= USE_AMP):
            d_real = crit_model(inputs, condition)
            d_gen = crit_model(generated.detach(), condition)
            with torch.cuda.amp.autocast(enabled= False):
                gp = gradient_penalty(critic_model= crit_model, real_data= inputs, generated_data= generated, condition= condition,\
                                    writer= writer)
            ### calculating the critic loss
            crit_loss = d_gen.mean() - d_real.mean() + gp + 2

        if USE_AMP:
            fabric.backward(scaler.scale(crit_loss))
        else:
            fabric.backward(crit_loss)  

        crit_train_avm.update(crit_loss.float().item())
        
        ## Get the current gradients and freeze it so we can load it back after the generator training step
        ## which requires backward through the critic
        ## Due to accumulated gradients steps and generator training steps is not the same, we can't just do crit_optimizer.zero_grad()
        ## before any critic training step and/or after any generator training step
        crit_model.get_grad()

        if step % ACCUMULATED_OPTIMIZER_STEP == 0:
            if USE_AMP:
                scaler.step(crit_optimizer)
                scaler.update()
            else:
                crit_optimizer.step()
            crit_optimizer.zero_grad()

    else:
        ########################
        ## GENERATOR TRAINING ##  (take 1*ACCUMULATED_OPTIMIZER_STEP step every ACCUMULATED_OPTIMIZER_STEP * GENERATOR_STEP_EVERY_N_CRITIC_STEP critic steps)
        ########################
        
        ## TODO: May be (idx + 1) % ACCUMULATED_OPTIMIZER_STEP * GENERATOR_STEP_EVERY_N_CRITIC_STEP == 0 is more accurate when talking to accumulated training,
        ## and then we can simutaneously train the generator for ACCUMULATED_OPTIMIZER_STEP and update that? (DONE!)
            # if step % GENERATOR_STEP_EVERY_N_CRITIC_STEP == 0:

        ## But that is quite hard to implement and we may need to store the data from last ACCUMULATED_OPTIMIZER_STEP of the critic training steps and then used them 
        ## for the training steps the generator

            
            ## create a generated batch 
            # generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
        with torch.cuda.amp.autocast(enabled = USE_AMP):
            ## Take the critic score again due to the generated images has already been *detach* from the generator
            d_gen1 = crit_model(generated, condition)
            
            ## Calculate the distribution of the generated data
            gen_mu, gen_log_var = gen_model.encoder(generated)
            ### VAE loss 
            rec_loss = loss_mse(inputs, generated).mean()
            
            ## For KL-loss, we replace the sum in log space with torch.mean for more stable training (or may not:)) )
            ## I suspect that the thing that make the generation loss explode in validation epoch is the score from critic,
            ## so I have add a tanh activation layer at the end of the critic to prevent this
            
            kl_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            ### Similarity loss (KL-Divergence between original and generated distribution)
            sim_loss = torch.mean(-0.5 * torch.mean(log_var - gen_log_var - ((gen_mu - mu) ** 2)/(gen_log_var.exp() + EPS) - (log_var - gen_log_var).exp(), dim = 1), dim = 0)

            gen_loss = (rec_loss + .2 * kl_loss + .2 * sim_loss) - d_gen1.mean() + 1

        if USE_AMP:
            fabric.backward(scaler.scale(gen_loss))
        else:
            fabric.backward(gen_loss)

        gen_train_avm.update(gen_loss.float().item())
        
        ## Load the original gradients of the critic after we backward the generator loss
        crit_model.load_grad()

        # update model parameters every n batches
        if (step + 1)  % ACCUMULATED_OPTIMIZER_STEP == 0:
            if USE_AMP:
                scaler.step(gen_optimizer)
                scaler.update()
                gen_optimizer.zero_grad()
            else:
                gen_optimizer.step()
                gen_optimizer.zero_grad()

        
    
    writer.add_scalars('loss/train_crit',{'val': crit_train_avm.val,'avg': crit_train_avm.avg})
    writer.add_scalars('loss/train_gen',{'val': gen_train_avm.val,'avg': gen_train_avm.avg})


    pbar.set_postfix({'crit_loss': crit_train_avm.avg, 'gen_loss': gen_train_avm.avg})

 

    # gen_scheduler.step()
    # crit_scheduler.step()

    
    # TODO: Change this from validation every n training epochs to every n steps (DONE!)
    if step % VAL_EPOCH_EVERY_N_TRAIN_STEPS == 0:

        gen_model.eval()
        crit_model.eval()
        # val_avm.reset()
        with torch.no_grad():
            for idx, batch in enumerate(pbar := tqdm(val_loader, position=0, leave= False, desc='Validating')):
                inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
                with torch.cuda.amp.autocast(enabled = USE_AMP):
                    ## create a generated batch 
                    generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
                    ### critic take action
                    d_real = crit_model(inputs, condition)
                    d_gen = crit_model(generated.detach(), condition)
    
                    gen_mu, gen_log_var = gen_model.encoder(generated.detach())
                    
                    
                    ## We ignore the gradient penalty in the evaluation process because the grad is diasbled here and through monitoring the training process,
                    ## we conclude that the gradient penalty is already close to 0
                    
                    # gp = gradient_penalty(critic_model= crit_model, real_data= inputs, generated_data= generated, condition= condition,\
                    #                     writer= writer)
    
                    ### Calculating the loss
    
                    crit_loss = d_gen.mean() - d_real.mean() + 2
                    
                    rec_loss = loss_mse(inputs, generated)
                    kl_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                    sim_loss = torch.mean(-0.5 * torch.mean(log_var - gen_log_var - ((gen_mu - mu) ** 2)/(gen_log_var.exp() + EPS) - (log_var - gen_log_var).exp(), dim = 1), dim = 0)
    
                    gen_loss = (rec_loss.mean() + .2 * kl_loss + .2 * sim_loss) - d_gen.mean() + 1
                    
                ## During ealier training, we see that the generator loss at validation step goes to infinity, so we want to check its components,
                ## although the suspection of that come from critic has already been fixed with adding a final tanh activation function.
                if idx in range(2,5):
                    print('\nReconstruction loss: ', rec_loss.item())
                    print('KL-Divergence loss: ', kl_loss.item())
                    print('Similarity loss: ', sim_loss)
                    print('Critic loss: ', -d_gen.mean().item())
                    print('Real loss', d_real.mean().item())
                    print('Fake loss', -d_gen.mean().item())
                
                exit()

                crit_val_avm.update(crit_loss.float().item())
                gen_val_avm.update(gen_loss.float().item())


                writer.add_scalars('loss/val_crit',{'val': crit_val_avm.val,'avg': crit_val_avm.avg})
                writer.add_scalars('loss/val_gen',{'val': gen_val_avm.val,'avg': gen_val_avm.avg})

                pbar.set_postfix({'crit_loss': crit_val_avm.avg, 'gen_loss': gen_val_avm.avg})
                
                
                # TODO: Changing this from fixed index to random index or even random samples across batches
                if (idx + 1) == 5:
                    writer.add_figure('sample_images',create_logging_plot(inputs.float(), condition.float(), generated.float()))   

        writer.add_scalars('loss/val_epoch',{'gen': gen_val_avm.avg,'crit': crit_val_avm.avg})



    if step % (SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS * VAL_EPOCH_EVERY_N_TRAIN_STEPS) == 0:
        for sub_dir in SUB_CKPT_DIR:
            if 'encoder' in sub_dir:
                save_checkpoint(sub_dir, model= gen_model.encoder, optimizer= gen_optimizer, scheduler= None, \
                                loss_meter= gen_val_avm,best_loss= gen_best_loss, step = step, writer = None)
                
            elif 'decoder' in sub_dir:
                loss = save_checkpoint(sub_dir, model= gen_model.decoder, optimizer= gen_optimizer, scheduler= None, \
                                loss_meter= gen_val_avm,best_loss= gen_best_loss, step = step, writer = None)
                gen_best_loss = gen_best_loss if loss is None else loss
            elif 'critic' in sub_dir:
                loss = save_checkpoint(sub_dir, model= crit_model, optimizer= crit_optimizer, scheduler= None, \
                                loss_meter= crit_val_avm,best_loss= crit_best_loss, step = step, writer = None)        
                crit_best_loss = crit_best_loss if loss is None else loss            
            
            




