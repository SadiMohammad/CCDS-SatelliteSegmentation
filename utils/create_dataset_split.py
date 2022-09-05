from sklearn.model_selection import train_test_split
import os
from natsort import natsorted

image_path = (
    "/media/sadi/Vol_2/IUB_research_local/Satellite/data/deepglobe_kaggle/archive/train"
)
data_split_dir = "/media/sadi/Vol_2/IUB_research_local/Satellite/CCDS-SatelliteSegmentation/dataloaders/deepglobe_split0"
files = os.listdir(image_path)
image_files = natsorted(list(filter(lambda k: "_sat" in k, files)))
label_files = natsorted(list(filter(lambda k: "_mask" in k, files)))

image_train, image_test, label_train, label_test = train_test_split(
    image_files, label_files, test_size=142, random_state=0
)
image_train, image_val, label_train, label_val = train_test_split(
    image_train, label_train, test_size=207, random_state=0
)

if not os.path.exists(data_split_dir):
    os.makedirs(data_split_dir)

with open(os.path.join(data_split_dir, "train_img_ids.txt"), "w") as f:
    for line in image_train:
        f.write(line)
        f.write("\n")

with open(os.path.join(data_split_dir, "train_label_ids.txt"), "w") as f:
    for line in label_train:
        f.write(line)
        f.write("\n")

with open(os.path.join(data_split_dir, "val_img_ids.txt"), "w") as f:
    for line in image_val:
        f.write(line)
        f.write("\n")

with open(os.path.join(data_split_dir, "val_label_ids.txt"), "w") as f:
    for line in label_val:
        f.write(line)
        f.write("\n")
with open(os.path.join(data_split_dir, "test_img_ids.txt"), "w") as f:
    for line in image_test:
        f.write(line)
        f.write("\n")
with open(os.path.join(data_split_dir, "test_label_ids.txt"), "w") as f:
    for line in label_test:
        f.write(line)
        f.write("\n")
