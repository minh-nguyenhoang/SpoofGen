import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from spoof_gen.data_utils import StandardDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from spoof_gen.models.generator import Encoder, Decoder, Generator
from spoof_gen.models.conditional_generator import ConditionGenerator
from spoof_gen.models.critic import Critic
from spoof_gen.utils import Contrast_depth_loss, AverageMeter, ExponentialMeter
import neptune
import torch.optim as opt
import lightning as L

from tqdm.auto import tqdm


fabric = L.Fabric(accelerator="cuda", devices=1, strategy="auto")
fabric.launch()
fabric.seed_everything(1273)

def create_logging_plot(inputs: torch.Tensor,condition: torch.Tensor, generated: torch.Tensor):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.cla()  # clear previous figure to free memory used
    plt.clf()
    N,_,_,_ = inputs.shape
    fig,ax = plt.subplots(3,N//2, figsize= (20,12))
    i = 0
    while i< N:
        idx = np.random.randint(0,N)
        ax[0,i].imshow(inputs[idx].permute(1,2,0).cpu().numpy());
        ax[0,i].axis('off');

        ax[1,i].imshow(condition[idx].cpu().numpy(), cmap= 'gray');
        ax[1,i].axis('off');

        ax[2,i].imshow(generated[idx].permute(1,2,0).cpu().numpy());
        ax[2,i].axis('off');

        i += 1

    return fig


##################################
####### GENERAL SETTINGS #########
##################################
VALIDATION_CHECKER = True


##################################
####### TRAINING SETTINGS ########
##################################
EPOCHS = 30
BATCH_SIZE = 1
ACCUMULATED_OPTIMIZER_STEP = 4
VAL_EPOCH_EVERY_N_TRAIN_EPOCHS = 1
SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS = 1

GENERATOR_STEP_EVERY_N_CRITIC_STEP = 5
GRADIENT_PENALTY_WEIGHT = 10

def gradient_penalty(critic_model, real_data, generated_data, condition, neptune_runner):
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
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.shape).to(device),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    if neptune_runner is not None:
        if critic_model.training:
            neptune_runner['monitoring/train_gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])
        else:
            neptune_runner['monitoring/val_gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return GRADIENT_PENALTY_WEIGHT * ((gradients_norm - 1) ** 2).mean()

##################################
###### CHECKPOINT DIRECTORY ######
##################################
CKPT_DIR = 'checkpoints/genetor'

if not CKPT_DIR.endswith('/'):
    CKPT_DIR += '/'

SUB_CKPT_DIR = ['critic', 'encoder', 'decoder']

for i in range(len(SUB_CKPT_DIR)):
    if not SUB_CKPT_DIR[i].endswith('/'):
        SUB_CKPT_DIR[i] += '/'

MODEL_NAME = {'critic/': 'critic','encoder/': 'encoder_arcface_50','decoder/':'decoder_revnet_18'}

SAVE_BEST = True

def save_checkpoint(sub_dir,*,
                    model, optimizer:torch.optim.Optimizer = None, scheduler:torch.optim.lr_scheduler._LRScheduler = None, 
                    loss_meter : AverageMeter|ExponentialMeter = None, best_loss = None, epoch:int = None, save_best = SAVE_BEST):
    assert not save_best or (loss_meter is not None and best_loss is not None and epoch is not None), 'If you want to save the best checkpoint, please provide a loss meter, current best loss and epoch/step!'

    if 'test' in sub_dir:
        sub_dir = 'encoder/'
        test = True
    else:
        test = False
    

    if save_best and loss_meter.avg < best_loss:
        best_loss = loss_meter.avg
        torch.save({'model_checkpoint': model.state_dict(),
                    'optimizer_checkpoint': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler_checkoint': scheduler.state_dict() if scheduler is not None else None,
                    'epoch': epoch,
                    'loss': loss_meter.avg,
                    }, f'{CKPT_DIR}{MODEL_NAME}_best_epoch={epoch}_loss={loss_meter.avg}.pth')
        
    torch.save({'model_checkpoint': model.state_dict(),
                'optimizer_checkpoint': optimizer.state_dict() if optimizer is not None else None,
                'scheduler_checkoint': scheduler.state_dict() if scheduler is not None else None,
                'epoch': epoch if epoch is not None else None,
                'loss': loss_meter.avg if loss_meter is not None else None,
                }, f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_last.pth')
    
    if test:
        import os
        os.remove(f'{CKPT_DIR}{sub_dir}{MODEL_NAME[sub_dir]}_last.pth')
        if save_best:
            os.remove(f'{CKPT_DIR}{MODEL_NAME}_best_epoch={epoch}_loss={loss_meter.avg}.pth')
    
    return best_loss


##################################
###### OPTIMIZER & SCHDULER ######
##################################
LR = 1e-3

STEP_SIZE = 10
GAMMA = 0.5



##################################
############## DATA ##############
##################################

transform = transforms.Compose([transforms.RandomRotation(30), transforms.RandomHorizontalFlip()])

train_data = StandardDataset('Data/train_img', transform= transform)
train_loader = DataLoader(train_data, batch_size= BATCH_SIZE, shuffle= True)

val_data = StandardDataset('Data/test_img', transform= None)
val_loader = DataLoader(val_data, batch_size= int(1.5*BATCH_SIZE), shuffle= False)


##################################
## MODEL, LOSS, OPTIMIZER, ECT. ##
##################################

encoder = Encoder(init_from='checkpoints/misc/backbone_arcface_50.pth')
decoder = Decoder(model= 'revnet-18')
con_gen = ConditionGenerator(init_from='checkpoints/conditional_generator/conditonal_generator_mobilenetv3_last.pth')
gen_model = Generator(encoder= encoder, decoder= decoder, con_gen= con_gen)

gen_optimizer = opt.Adam(gen_model.parameters(),lr = LR)
gen_scheduler = opt.lr_scheduler.StepLR(optimizer= gen_optimizer, step_size= STEP_SIZE, gamma= GAMMA)


crit_model = Critic()

crit_optimizer = opt.Adam(crit_model.parameters(),lr = LR)
crit_scheduler = opt.lr_scheduler.StepLR(optimizer= crit_optimizer, step_size= STEP_SIZE, gamma= GAMMA)

loss_mse = nn.MSELoss()


##################################
##### SETUP LIGHTNING FABRIC #####
##################################
gen_model, gen_optimizer, gen_scheduler = fabric.setup(gen_model, gen_optimizer, gen_scheduler)
crit_model, crit_optimizer, crit_scheduler = fabric.setup(crit_model, crit_optimizer, crit_scheduler)

train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

##################################
####### SETUP LOSS METER #########
##################################

## The use of exponential smoothing over averaging for easier weighting of current value

crit_best_loss = torch.inf
gen_best_loss = torch.inf
gen_train_avm = ExponentialMeter()
gen_val_avm = ExponentialMeter()
crit_train_avm = ExponentialMeter()
crit_val_avm = ExponentialMeter()


##################################
#### TRAINING AND VALIDATING #####
##################################

## Validation checker

if VALIDATION_CHECKER:
    BREAK_POINT = 10

    check_loader = DataLoader([val_data[i] for i in range(BREAK_POINT * int(1.5*BATCH_SIZE))], batch_size= int(1.5*BATCH_SIZE), shuffle= False)
    check_loader = fabric.setup_dataloaders(check_loader)
    try:
        print("Start checking")
        for idx, batch in enumerate(pbar := tqdm(check_loader, desc='Validation checker', leave= False)):
            inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
            ## create a generated batch 
            generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
            ### critic take action
            d_real = crit_model(inputs, condition)
            d_gen = crit_model(generated.detach(), condition)

            gen_mu, gen_log_var = gen_model.encoder(generated.detach())
        
            gp = gradient_penalty(critic_model= crit_model, real_data= inputs, generated_data= generated, condition= condition,\
                                neptune_runner= None)

            ### calculating the loss

            crit_loss = d_gen.mean() - d_real.mean() + gp

            fabric.backward(crit_loss)
            
            crit_optimizer.zero_grad()

            # generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
            d_gen1 = crit_model(generated, condition)

            rec_loss = loss_mse(inputs, generated)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            sim_loss = torch.mean(-0.5 * torch.sum(log_var - gen_log_var - ((gen_mu - mu) ** 2)/gen_log_var.exp() - (log_var - gen_log_var).exp(), dim = 1), dim = 0)
            gen_loss = (rec_loss.mean() + .2 * kl_loss + .2 * sim_loss) - d_gen1.mean()

            fabric.backward(gen_loss)
            gen_optimizer.zero_grad()

            # input('Press Enter!')

            pbar.set_postfix({'crit_loss': crit_loss, 'gen_loss': gen_loss})

        ## test save

        loss = save_checkpoint('test/', model= gen_model.encoder, optimizer= gen_optimizer, scheduler= gen_scheduler, \
                                    loss_meter= gen_val_avm ,best_loss= gen_best_loss, epoch= -1, save_best= True)
            
        ## test plot
        create_logging_plot(inputs, condition, generated)


    except Exception as e:
        print('>>> '+ e)
        exit()

    # finally:
    #     ## just for checking, remove this when training

    #     raw = input("Seem OK, press Enter to exit")
    #     exit()


raw = input("Seem OK, press Enter to exit")
exit()
##################################
###### NEPTUNE AI LOGGER #########
##################################
RUN_ID = None

if RUN_ID is None:
    run = neptune.init_run(
        project="minhnguyen/Generate-spoof-from-real-images",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNThiYzVhMy0xYzM4LTQ5ZmItOWFkZC00YmIzODljNzM1MmUifQ==",
        # with_id=
    )  
else:
    try:
        run = neptune.init_run(
            project="minhnguyen/Generate-spoof-from-real-images",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNThiYzVhMy0xYzM4LTQ5ZmItOWFkZC00YmIzODljNzM1MmUifQ==",
            with_id= f"GEN-{RUN_ID}"
    )  
    except Exception as e:
        # raise ValueError("The RUN_ID provided is invalid")
        print(e)
        exit()


optimizer_params = { "optimizer": "Adam", "learning_rate": LR, 'scheduler': "StepLR", 'step_size': STEP_SIZE, 'gamma': GAMMA}
run["parameters"] = optimizer_params



for epoch in (ep_bar := tqdm(range(1,EPOCHS+1))):
    ep_bar.set_description(f'Epoch {epoch}/{EPOCHS}')

    gen_model.train()
    crit_model.train()
    for idx, batch in enumerate(pbar := tqdm(train_loader, leave= False, desc='Training')):
        inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
        
        
        ## create a generated batch 
        generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)

        ### deatch the generated images so the grad of generator will not be update here
        generated = generated.detach()

        #####################
        ## CRITIC TRAINING ##
        #####################

        ### critic take action
        d_real = crit_model(inputs, condition)
        d_gen = crit_model(generated, condition)

        gp = gradient_penalty(critic_model= crit_model, real_data= inputs, generated_data= generated, condition= condition,\
                              neptune_runner= run)

        ### calculating the critic loss

        crit_loss = d_gen.mean() - d_real.mean() + gp


        fabric.backward(crit_loss)

        crit_train_avm.update(crit_loss)

        run['loss/train_cit'].append(crit_loss)
        run['loss/train_cit_avg'].append(crit_train_avm.avg)

        if (idx + 1) % ACCUMULATED_OPTIMIZER_STEP == 0:
            crit_optimizer.step()
            crit_optimizer.zero_grad()

        ########################
        ## GENERATOR TRAINING ##  (take a step every n critic steps)
        ########################

        if (idx + 1) % GENERATOR_STEP_EVERY_N_CRITIC_STEP:
            ## create a generated batch 
            generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
            d_gen = crit_model(generated, condition)

            gen_mu, gen_log_var = gen_model.encoder(generated.detach())
            ### VAE loss 
            rec_loss = loss_mse(inputs, generated)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            ### Similarity loss (KL-Divergence between original and generated distribution)
            sim_loss = torch.mean(-0.5 * torch.sum(log_var - gen_log_var - ((gen_mu - mu) ** 2)/gen_log_var.exp() - (log_var - gen_log_var).exp(), dim = 1), dim = 0)

            gen_loss = (rec_loss.mean() + .2 * kl_loss + .2 * sim_loss) - d_gen.mean()


            fabric.backward(gen_loss)
            ### clear the gradients for critic
            crit_optimizer.zero_grad()

            gen_train_avm.update(gen_loss)

            run['loss/train_gen'].append(gen_loss)
            run['loss/train_gen_avg'].append(gen_train_avm.avg)

            # update model parameters every n batches
            if (idx + 1) % ACCUMULATED_OPTIMIZER_STEP * GENERATOR_STEP_EVERY_N_CRITIC_STEP == 0:
                gen_optimizer.step()
                gen_optimizer.zero_grad()


        pbar.set_postfix({'crit_loss': crit_train_avm.avg, 'gen_loss': gen_train_avm.avg})


    # update model parameters at the end of the epoch if the final batch has smaller size than ACCUMULATED_OPTIMIZER_STEP
    gen_optimizer.step()
    gen_optimizer.zero_grad()

    crit_optimizer.step()
    crit_optimizer.zero_grad()
    

    gen_scheduler.step()
    crit_scheduler.step()


    if epoch % VAL_EPOCH_EVERY_N_TRAIN_EPOCHS == 0:
        gen_model.train()
        crit_model.train()
        # val_avm.reset()
        with torch.no_grad():
            for idx, batch in enumerate(pbar := tqdm(val_loader, leave= False, desc='Validating')):
                inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
                ## create a generated batch 
                generated, condition, (mu, log_var) = gen_model(inputs, spoof_label)
                ### critic take action
                d_real = crit_model(inputs, condition)
                d_gen = crit_model(generated, condition)

                gen_mu, gen_log_var = gen_model.encoder(generated.detach())
            
                gp = gradient_penalty(critic_model= crit_model, real_data= inputs, generated_data= generated, condition= condition,\
                                    neptune_runner= run)

                ### calculating the loss

                crit_loss = d_gen.mean() - d_real.mean() + gp

                rec_loss = loss_mse(inputs, generated)
                kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                sim_loss = torch.mean(-0.5 * torch.sum(log_var - gen_log_var - ((gen_mu - mu) ** 2)/gen_log_var.exp() - (log_var - gen_log_var).exp(), dim = 1), dim = 0)

                gen_loss = (rec_loss.mean() + .2 * kl_loss + .2 * sim_loss) - d_gen.mean()

                crit_val_avm.update(crit_loss)
                gen_val_avm.update(gen_loss)

                run['loss/val_crit'].append(crit_loss)
                run['loss/val_crit_avg'].append(crit_val_avm.avg)

                run['loss/val_gen'].append(gen_loss)
                run['loss/val_gen_avg'].append(gen_val_avm.avg)

                pbar.set_postfix({'crit_loss': crit_val_avm.avg, 'gen_loss': gen_val_avm.avg})

                if idx == 5:
                    run['sample_images'].append(create_logging_plot(inputs, condition, generated))   
    
        if epoch % (SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS * VAL_EPOCH_EVERY_N_TRAIN_EPOCHS) == 0:
            for sub_dir in SUB_CKPT_DIR:
                if 'encoder' in sub_dir:
                    loss = save_checkpoint(sub_dir, model= gen_model.encoder, optimizer= gen_optimizer, scheduler= gen_scheduler, \
                                    loss_meter= gen_val_avm,best_loss= gen_best_loss, epoch= epoch)
                    gen_best_loss = gen_best_loss if loss is None else loss
                elif 'decoder' in sub_dir:
                    save_checkpoint(sub_dir, model= gen_model.decoder, optimizer= gen_optimizer, scheduler= gen_scheduler, \
                                    loss_meter= gen_val_avm,best_loss= gen_best_loss, epoch= epoch)
                elif 'critic' in sub_dir:
                    loss = save_checkpoint(sub_dir, model= crit_model, optimizer= crit_optimizer, scheduler= crit_scheduler, \
                                    loss_meter= crit_val_avm,best_loss= crit_best_loss, epoch= epoch)        
                    crit_best_loss = crit_best_loss if loss is None else loss            
            
            
    ep_bar.set_postfix({'train_gen_loss': gen_train_avm.avg, 'val_gen_loss': gen_val_avm.avg,
                        'train_crit_loss': crit_train_avm.avg, 'val_crit_loss': crit_val_avm.avg})



