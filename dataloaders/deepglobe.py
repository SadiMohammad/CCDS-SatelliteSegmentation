import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class Dataset_ROM(Dataset):
    def __init__(self, cfgs, transformers):
        self.cfgs = cfgs
        self.transformers = transformers
        df = pd.read_csv(cfgs["dataset"]["image_ids"], header=None)
        self.image_files = df[df.columns[0]].tolist()
        df = pd.read_csv(cfgs["dataset"]["gt_ids"], header=None)
        self.gt_files = df[df.columns[0]].tolist()

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.cfgs["dataset"]["image_path"], self.image_files[idx])
        ).convert(self.cfgs["dataset"]["img_convert"])
        gt = Image.open(
            os.path.join(self.cfgs["dataset"]["gt_path"], self.gt_files[idx])
        ).convert(self.cfgs["dataset"]["gt_convert"])
        if self.transformers:
            t_image = self.transformers["image"](image)
            if "train" in self.cfgs["experiment_name"].lower():
                t_gt = self.transformers["gt"](gt)
        return t_image, t_gt

    def __len__(self):
        return len(self.image_files)