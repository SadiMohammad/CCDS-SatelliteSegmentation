import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np


def get_color_map(class_dict_path):
    class_dict = pd.read_csv(os.path.join(class_dict_path, "class_dict.csv"))
    num_class = len(class_dict)
    colors = []
    for (r, g, b) in class_dict[["r", "g", "b"]].to_numpy():
        colors.append([r, g, b])
    map_color = {x: v for x, v in zip(range(num_class), colors)}
    return map_color


def rgb_to_onehot(rgb_arr, color_dict):
    assert len(rgb_arr.shape) == 3
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2] + (num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls_ in enumerate(color_dict):
        arr[:, :, i] = np.all(
            rgb_arr.reshape((-1, 3)) == color_dict[i], axis=1
        ).reshape(shape[:2])
    return arr


class DeepGlobe_ROM(Dataset):
    def __init__(self, cfgs, subset, transformers):
        self.cfgs = cfgs
        self.subset = subset
        self.transformers = transformers
        self.map_color = get_color_map(self.cfgs["dataset"]["class_dict_path"])
        df = pd.read_csv(cfgs["dataset"][self.subset + "_image_ids"], header=None)
        self.image_files = df[df.columns[0]].tolist()
        df = pd.read_csv(cfgs["dataset"][self.subset + "_label_ids"], header=None)
        self.label_files = df[df.columns[0]].tolist()

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.cfgs["dataset"]["full_image_path"], self.image_files[idx])
        ).convert(self.cfgs["dataset"]["img_convert"])
        label = Image.open(
            os.path.join(self.cfgs["dataset"]["full_label_path"], self.label_files[idx])
        ).convert(self.cfgs["dataset"]["label_convert"])
        if self.transformers:
            t_image = self.transformers["image"](image)
            if "train" in self.cfgs["experiment_name"].lower():
                t_label = self.transformers["label"](label)

        patch_images = []
        patch_labels = []
        for patch_id in range(self.cfgs["dataset"]["no_of_patches"]):
            patch_image_file = "{}_{}.{}".format(
                self.image_files[idx].split(".")[0],
                str(patch_id),
                self.image_files[idx].split(".")[1],
            )
            patch_image = Image.open(
                os.path.join(
                    self.cfgs["dataset"]["patches_image_path"], patch_image_file
                )
            ).convert(self.cfgs["dataset"]["img_convert"])
            patch_label_file = "{}_{}.{}".format(
                self.label_files[idx].split(".")[0],
                str(patch_id),
                self.label_files[idx].split(".")[1],
            )
            patch_label = Image.open(
                os.path.join(
                    self.cfgs["dataset"]["patches_label_path"], patch_label_file
                )
            ).convert(self.cfgs["dataset"]["label_convert"])
            if self.transformers:
                t_patch_image = self.transformers["image"](image)
                if "train" in self.cfgs["experiment_name"].lower():
                    t_patch_label = self.transformers["label"](label)
            patch_images.append(t_patch_image)
            patch_labels.append(t_patch_label)
            t_patch_images = torch.stack(patch_images)
            t_patch_labels = torch.stack(patch_labels)

        return t_image, t_label, t_patch_images, t_patch_labels

    def __len__(self):
        return len(self.image_files)
