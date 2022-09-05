from unittest.mock import patch
from dataloaders.deepglobe import DeepGlobe_ROM
from configs.config import Config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import torch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--config_file",
    help="config file name/experiment name",
    default="train",
)
args = arg_parser.parse_args()
Cfgs = Config(args)
cfgs = Cfgs.cfgs

# DATA LOADERS
transformers = {
    "image": transforms.Compose(
        [
            transforms.Resize(
                (cfgs["dataset"]["input_size"], cfgs["dataset"]["input_size"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "label": transforms.Compose(
        [
            transforms.Resize(
                (cfgs["dataset"]["input_size"], cfgs["dataset"]["input_size"])
            ),
            transforms.ToTensor(),
        ]
    ),
}

dataset_train = DeepGlobe_ROM(cfgs, subset="train", transformers=transformers)
loader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
    pin_memory=True,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i_train, sample_train in enumerate(tqdm(loader_train)):
    images = sample_train[0].to(device)
    labels = sample_train[1].to(device)
    patched_images = sample_train[2].to(device)
    patched_labels = sample_train[3].to(device)
    print(patched_images[:,0,:,:,:].size())
    print('s')
