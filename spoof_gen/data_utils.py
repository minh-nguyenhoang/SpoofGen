import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import random


class StandardDataset(Dataset):
    def __init__(self, root_dir='',  transform=None, preload=True, **kwarg):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess()
        if preload:
            with open(f'./{self.root_dir}Imfile.txt', 'r+') as f:
                lines = f.readlines()
                self.lines = [path.strip().split(" ") for path in lines]
            os.remove(f'./{self.root_dir.upper()}Imfile.txt')
        else:
            self.lines = None

    def __len__(self):
        if self.lines is not None:
            return len(self.lines)

        with open(f'./{self.root_dir}Imfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
        return len_ds-1

    def __getitem__(self, idx):

        if self.lines is not None:
            dir, filename = self.lines[idx][0], self.lines[idx][1]
        else:
            with open(f'./{self.root_dir}Imfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename = lines[idx].strip().split(" ")

        ## only choosing real face with depth map
        # while 'fake' in dir+filename:
        #     idx = random.randint(0, self.__len__()-1)
        #     if self.lines is not None:
        #         dir, filename = self.lines[idx][0], self.lines[idx][1]
        #     else:
        #         with open(f'./{self.root_dir.upper()}Imfile.txt', 'r+') as f:
        #             lines = f.readlines()
        #         dir, filename = lines[idx].strip().split(" ")


        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        try:
            depth_im = torch.Tensor(cv2.resize(cv2.imread(dir+'/depth'+filename, 0),
                                (256, 256), interpolation=cv2.INTER_CUBIC)).unsqueeze(-1).permute(2, 0, 1)/255
        except:
            depth_im = torch.zeros((256,256)).unsqueeze(-1).permute(2, 0, 1)/255



        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            # sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])
            sample[:3] = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(sample[:3])
            

        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):
        with open(f'./{self.root_dir}Imfile.txt', 'w+') as f:
            # self.data = {'color':[] , 'depth':[]}
            dir = self.root_dir + r'/color'
            for filename in os.listdir(dir):
                f.writelines(self.root_dir + f' /{filename}'+'\n')


class CombinedDataset(Dataset):
    def __init__(self, root_dir:list[str]= [''],  transform=None, preload=True, **kwarg):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess()
        if preload:
            with open(f'./{self.filename.upper()}Imfile.txt', 'r+') as f:
                lines = f.readlines()
                self.lines = [path.strip().split(" ") for path in lines]
            os.remove(f'./{self.filename.upper()}Imfile.txt')
        else:
            self.lines = None

    def __len__(self):
        if self.lines is not None:
            return len(self.lines)

        with open(f'./{self.filename.upper()}Imfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
            return len_ds-1

    def __getitem__(self, idx):
        if self.lines is not None:
            dir, filename = self.lines[idx][0], self.lines[idx][1]
        else:
            with open(f'./{self.filename.upper()}Imfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename = lines[idx].strip().split(" ")

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.resize(cv2.imread(dir+'/depth'+filename, 0),
                                (256, 256), interpolation=cv2.INTER_CUBIC)).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            # sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])
            sample[:3] = transforms.ColorJitter(
                brightness=0.4, contrast=0.3, saturation=0.2, hue=0.1)(sample[:3])
        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):

        self.filename = ''
        for dir in self.root_dir:
            self.filename += dir.replace('/','_')
        with open(f'./{self.filename.upper()}Imfile.txt', 'w+') as f:
            for dir in self.root_dir:
                dir = dir + r'/color'
                for file in os.listdir(dir):
                    f.writelines(dir + f' /{file}'+'\n')

