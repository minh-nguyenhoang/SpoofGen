import torch
import torch.nn as nn
from data_utils import StandardDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from VAEGAN.models.conditional_generator import ConditionGenerator
from VAEGAN.utils import Contrast_depth_loss, AverageMeter
import neptune.new as neptune
import torch.optim as opt
import lightning as L

from tqdm.auto import tqdm


fabric = L.Fabric(accelerator="cuda", devices=1, strategy="auto")
fabric.launch()



##################################
####### TRAINING SETTINGS ########
##################################
EPOCHS = 30
BATCH_SIZE = 12
ACCUMULATED_OPTIMIZER_STEP = 1
VAL_EPOCH_EVERY_N_TRAIN_EPOCHS = 1
SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS = 1


##################################
#### CHECKPOINT DIRECTORY ########
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
############## DATA ##############
##################################
transform = transforms.Compose([transforms.RandomRotation(.15), transforms.RandomHorizontalFlip()])

train_data = StandardDataset('Data/train_img', transform= transform)
train_loader = DataLoader(train_data, batch_size= BATCH_SIZE, shuffle= True)

val_data = StandardDataset('Data/test_img', transform= None)
val_loader = DataLoader(val_data, batch_size= int(1.5*BATCH_SIZE), shuffle= False)


##################################
## MODEL, LOSS, OPTIMIZER, ECT. ##
##################################

model = ConditionGenerator().cuda()

optimizer = opt.Adam(model.parameters(),lr = LR)
scheduler = opt.lr_scheduler.StepLR(optimizer= optimizer, step_size= STEP_SIZE, gamma= GAMMA)


loss_mse = nn.MSELoss()
loss_cdl = Contrast_depth_loss()


##################################
##### SETUP LIGHTNING FABRIC #####
##################################
model, optimizer, loss_cdl, scheduler = fabric.setup(model, optimizer, loss_cdl, scheduler)
train_loader, val_laoder = fabric.setup_dataloaders(train_loader, val_loader)


##################################
###### NEPTUNE AI LOGGER #########
##################################
run = neptune.init_run(
    project="minhnguyen/Generate-spoof-from-real-images",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNThiYzVhMy0xYzM4LTQ5ZmItOWFkZC00YmIzODljNzM1MmUifQ==",
    # with_id=
)  

optimizer_params = {"learning_rate": LR, "optimizer": "Adam", 'step_size': STEP_SIZE, 'gamma': GAMMA}
run["parameters"] = optimizer_params



##################################
#### TRAINING AND VALIDATING #####
##################################

best_loss = torch.inf
train_avm = AverageMeter()
val_avm = AverageMeter()

for epoch in (ep_bar := tqdm(range(1,EPOCHS+1))):
    ep_bar.set_description(f'Epoch {epoch}')

    model.train()
    for idx, batch in enumerate(tqdm(train_loader, leave= False, desc='Training')):
        inputs, map_label, spoof_label = batch[0].float().cuda(), batch[1].float().cuda(), batch[2].float().cuda()
        map_x = model(batch,1)
        loss = loss_mse(map_label, map_x) + loss_cdl(map_label, map_x)
        # loss.backward()
        fabric.backward(loss)

        train_avm.update(loss)

        run['loss/train'].append(loss)
        run['loss/train_avg'].append(train_avm.avg)

        # update model parameters every n batches
        if (idx + 1) % ACCUMULATED_OPTIMIZER_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()

    # update model parameters at the end of the epoch
    optimizer.step()
    optimizer.zero_grad()
    

    scheduler.step()


    if epoch % VAL_EPOCH_EVERY_N_TRAIN_EPOCHS == 0:
        model.eval()
        val_avm.reset()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader, leave= False, desc='Validating')):
                inputs, map_label, spoof_label = batch[0].float(), batch[1].float(), batch[2].float()
                map_x = model(batch,1)
                loss = loss_mse(map_label, map_x) + loss_cdl(map_label, map_x)
                val_avm.update(loss)

                run['loss/val'].append(loss)
                run['loss/val_avg'].append(val_avm.avg)
    
        if epoch % (SAVE_CHECKPOINT_EVERY_N_VAL_EPOCHS * VAL_EPOCH_EVERY_N_TRAIN_EPOCHS) == 0:
            
            if SAVE_BEST and val_avm.avg < best_loss:
                best_loss = val_avm.avg
                torch.save({'model_checkpoint': model.state_dict(),
                            'optimizer_checkpoint': optimizer.state_dict(),
                            'scheduler_checkoint': scheduler.state_dict(),
                            'epoch': epoch,
                            'loss': val_avm.avg,
                            }, f'{CKPT_DIR}{MODEL_NAME}:best.pth')
                
            torch.save({'model_checkpoint': model.state_dict(),
                        'optimizer_checkpoint': optimizer.state_dict(),
                        'scheduler_checkoint': scheduler.state_dict(),
                        'epoch': epoch,
                        'loss': val_avm.avg,
                        }, f'{CKPT_DIR}{MODEL_NAME}:last.pth')

