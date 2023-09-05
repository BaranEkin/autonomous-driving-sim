import os
import random
import cv2
import torch
import torchvision.transforms as tr

from torch.utils.data import Dataset


class SimDataset(Dataset):
    def __init__(self, data_path):

        self.folder_dir = os.path.dirname(data_path)
        with open(data_path, 'r') as data_file:
            self.lines = data_file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        folder_name = self.lines[idx][1:4]
        img_path, throttle, brake, steering, velocity = self.lines[idx][4:].strip('\n').replace(',', '.').split(' ')

        img = cv2.imread(os.path.join(self.folder_dir, folder_name, img_path), flags=cv2.IMREAD_COLOR).astype(float)
        img_tensor = torch.autograd.Variable(torch.FloatTensor(img))
        img_mean = int(img_tensor.mean().item())
        transform = tr.Compose([tr.RandomErasing(p=0.3, scale=(.008, .08), ratio=(1, 2.5), value="random"),
                                tr.RandomErasing(p=0.3, scale=(.008, .08), ratio=(1, 2.5), value=img_mean),
                                tr.RandomErasing(p=0.3, scale=(.008, .08), ratio=(1, 2.5), value=255),
                                tr.RandomErasing(p=0.3, scale=(.008, .08), ratio=(1, 2.5), value=0)])

        return transform(img_tensor), float(velocity), float(throttle) - float(brake), float(steering),
