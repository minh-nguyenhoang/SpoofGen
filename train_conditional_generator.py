import torch
import torch.nn as nn
from spoof_gen.data_utils import StandardDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from spoof_gen.models.conditional_generator import ConditionGenerator
from spoof_gen.utils import Contrast_depth_loss, AverageMeter
import neptune
import torch.optim as opt
import lightning as L

from tqdm.auto import tqdm


fabric = L.Fabric(accelerator="cuda", devices=1, strategy="auto")
fabric.launch()
fabric.seed_everything(1273)

def create_logging_plot(inputs: torch.Tensor, map_x, map_label):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.cla()  # clear previous figure to free memory used
    plt.clf()
    N,_,_,_ = inputs.shape
    fig,ax = plt.subplots(3,N//2, figsize= (20,12))
    i = 0
    while i< N//2:
        idx = np.random.randint(0,N)
        ax[0,i].imshow(inputs[idx].permute(1,2,0).cpu().numpy());
        ax[0,i].axis('off');
        # ax[1,i].imshow(used_dataset[idx][1].numpy(), cmap= 'gray');
        # ax[1,i].axis('off');
        ax[1,i].imshow(map_x[idx].cpu().numpy(), cmap= 'gray');
        ax[1,i].axis('off');
        # grayscale_cam = (cam_cls(input_tensor=used_dataset[idx][0].unsqueeze(0).cuda(), targets=targets) + cam_net(input_tensor=used_dataset[idx][0].unsqueeze(0).cuda(), targets=targets))/2
        # grayscale_cam = cam_cls(input_tensor=used_dataset[idx][0].unsqueeze(0).cuda(), targets=targets)
        ax[2,i].imshow(map_label[idx].cpu().numpy(), cmap= 'gray');
        ax[2,i].axis('off');

        i += 1

    return fig

##################################
####### GENERAL SETTINGS #########
##################################
VALIDATE_CHECKER = False


##################################
####### TRAINING SETTINGS ########
##################################
EPOCHS = 30
BATCH_SIZE = 12
ACCUMULATED_OPTIMIZER_STEP = 1
VAL_EPOCH_EVERY_N_TRAIN_EPOCHS = 1
SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS = 1


##################################
###### CHECKPOINT DIRECTORY ######
##################################
CKPT_DIR = 'checkpoints/conditional_generator'

if not CKPT_DIR.endswith('/'):
    CKPT_DIR += '/'

MODEL_NAME = 'conditonal_generator_mobilenetv3'

SAVE_BEST = False

##################################
###### OPTIMIZER & SCHDULER ######
##################################
LR = 1e-3

STEP_SIZE = 10
GAMMA = 0.5


##################################
###### NEPTUNE AI LOGGER #########
##################################
RUN_ID = 1

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

model = ConditionGenerator()

optimizer = opt.Adam(model.parameters(),lr = LR)
scheduler = opt.lr_scheduler.StepLR(optimizer= optimizer, step_size= STEP_SIZE, gamma= GAMMA)


loss_mse = nn.MSELoss()
loss_cdl = Contrast_depth_loss()


##################################
##### SETUP LIGHTNING FABRIC #####
##################################
model, optimizer, scheduler = fabric.setup(model, optimizer, scheduler)
train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)



##################################
#### TRAINING AND VALIDATING #####
##################################

best_loss = torch.inf
train_avm = AverageMeter()
val_avm = AverageMeter()

for epoch in (ep_bar := tqdm(range(1,EPOCHS+1))):
    ep_bar.set_description(f'Epoch {epoch}/{EPOCHS}')

    model.train()
    for idx, batch in enumerate(pbar := tqdm(train_loader, leave= False, desc='Training')):
        inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
        map_x = model(inputs,1)
        loss = loss_mse(map_label, map_x) + loss_cdl(map_label, map_x)
        # loss.backward()
        fabric.backward(loss)

        train_avm.update(loss)

        run['loss/train'].append(loss)
        run['loss/train_avg'].append(train_avm.avg)

        pbar.set_postfix({'loss': loss, 'epoch_loss': train_avm.avg})
        # update model parameters every n batches
        if (idx + 1) % ACCUMULATED_OPTIMIZER_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()

    # update model parameters at the end of the epoch if the final batch has smaller size than ACCUMULATED_OPTIMIZER_STEP
    optimizer.step()
    optimizer.zero_grad()
    

    scheduler.step()


    if epoch % VAL_EPOCH_EVERY_N_TRAIN_EPOCHS == 0:
        model.eval()
        # val_avm.reset()
        with torch.no_grad():
            for idx, batch in enumerate(pbar := tqdm(val_loader, leave= False, desc='Validating')):
                inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
                map_x = model(inputs,1)
                loss = loss_mse(map_label, map_x) + loss_cdl(map_label, map_x)
                val_avm.update(loss)

                run['loss/val'].append(loss)
                run['loss/val_avg'].append(val_avm.avg)

                pbar.set_postfix({'loss': loss, 'epoch_loss': val_avm.avg})

                if idx == 5:
                    run['sample_images'].append(create_logging_plot(inputs, map_x, map_label))   
    
        if epoch % (SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS * VAL_EPOCH_EVERY_N_TRAIN_EPOCHS) == 0:
            
            if SAVE_BEST and val_avm.avg < best_loss:
                best_loss = val_avm.avg
                torch.save({'model_checkpoint': model.state_dict(),
                            'optimizer_checkpoint': optimizer.state_dict(),
                            'scheduler_checkoint': scheduler.state_dict(),
                            'epoch': epoch,
                            'loss': val_avm.avg,
                            }, f'{CKPT_DIR}{MODEL_NAME}_best.pth')
                
            torch.save({'model_checkpoint': model.state_dict(),
                        'optimizer_checkpoint': optimizer.state_dict(),
                        'scheduler_checkoint': scheduler.state_dict(),
                        'epoch': epoch,
                        'loss': val_avm.avg,
                        }, f'{CKPT_DIR}{MODEL_NAME}_last.pth')
            
    ep_bar.set_postfix({'train_loss': train_avm.avg, 'val_loss': val_avm.avg})



