from patchify import patchify
import os, sys
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

ids_path = "/media/sadi/Vol_2/IUB_research_local/Satellite/CCDS-SatelliteSegmentation/dataloaders/deepglobe_split0"
image_path = (
    "/media/sadi/Vol_2/IUB_research_local/Satellite/data/deepglobe_kaggle/archive/train"
)
patches_save_path = "/media/sadi/Vol_2/IUB_research_local/Satellite/data/deepglobe_kaggle/archive/patches/mask"
df_image = pd.read_csv(os.path.join(ids_path, "train_label_ids.txt"), header=None)
image_files = df_image[df_image.columns[0]].tolist()

if not os.path.exists(patches_save_path):
    os.makedirs(patches_save_path)
for file in tqdm(image_files):
    image = Image.open(os.path.join(image_path, file)).convert("RGB")
    image = image.resize((2048, 2048))
    image = np.array(image)
    patches = patchify(image, (512, 512, 3), step=512)

    patch_id = 0
    for i in range(0, patches.shape[0]):
        for j in range(0, patches.shape[1]):
            im = Image.fromarray(patches[i, j, 0, :, :, :])
            im.save(
                os.path.join(
                    patches_save_path, file.split(".")[0] + "_" + str(patch_id)
                )
                + "."
                + file.split(".")[1]
            )
            patch_id += 1
