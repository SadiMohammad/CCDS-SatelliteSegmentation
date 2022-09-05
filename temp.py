# %% one_hot_encode for dataloader
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
def get_color_map(base_dir):
    class_dict = pd.read_csv(os.path.join(base_dir, 'class_dict.csv'))
    num_class = len(class_dict)
    colors = []
    for (r,g,b) in class_dict[['r', 'g', 'b']].to_numpy():
        colors.append([r,g,b])
    map_color = {x:v for x,v in zip(range(num_class),colors)}
    return map_color

def rgb_to_onehot(rgb_arr, color_dict):
    assert len(rgb_arr.shape) == 3
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls_ in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

map_color = get_color_map('/media/sadi/Vol_2/IUB_research_local/Satellite/data/deepglobe_kaggle/archive')
cv_im = cv2.imread('/media/sadi/Vol_2/IUB_research_local/Satellite/data/deepglobe_kaggle/archive/train/119_mask.png', 1)
full_mask = cv2.resize(cv_im, (512,512), cv2.INTER_NEAREST)
image = Image.open('/media/sadi/Vol_2/IUB_research_local/Satellite/data/deepglobe_kaggle/archive/train/119_mask.png').convert('RGB')
# image = image.resize((512, 512))
image_arr = np.array(image)
x = rgb_to_onehot(image_arr, map_color)
y = np.argmax(x, axis=-1)
f=torch.from_numpy(y)[None, :]
t_f=torch.argmax(F.one_hot(f, num_classes=7),dim=3)
t=transforms.Resize(512)
resized_t_f = t(t_f)
print('s')

# # %% dataloader output check

# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import pandas as pd


# class Dataset_ROM(Dataset):
#     def __init__(self, cfgs, transformers):
#         self.cfgs = cfgs
#         self.transformers = transformers
#         df = pd.read_csv(cfgs["dataset"]["image_ids"], header=None)
#         self.image_files = df[df.columns[0]].tolist()
#         df = pd.read_csv(cfgs["dataset"]["gt_ids"], header=None)
#         self.gt_files = df[df.columns[0]].tolist()

#     def __getitem__(self, idx):
#         image = Image.open(
#             os.path.join(self.cfgs["dataset"]["image_path"], self.image_files[idx])
#         ).convert(self.cfgs["dataset"]["img_convert"])
#         gt = Image.open(
#             os.path.join(self.cfgs["dataset"]["gt_path"], self.gt_files[idx])
#         ).convert(self.cfgs["dataset"]["gt_convert"])
#         if self.transformers:
#             t_image = self.transformers["image"](image)
#             if "train" in self.cfgs["experiment_name"].lower():
#                 t_gt = self.transformers["gt"](gt)
#         return t_image, t_gt

#     def __len__(self):
#         return len(self.image_files)