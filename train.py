import time
import torch
import os, sys, random, string
import argparse
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms
from models.model import Model
from configs.config import Config
from utils.logger import Logger
from utils.transforms import *
from utils.metrics import eval_metrics
from utils.losses import CE_loss
from dataloaders.deepglobe import DeepGlobe_ROM
from trainer import Trainer

time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(Cfgs):
    cfgs = Cfgs.cfgs
    device = torch.device(cfgs["train_setup"]["device"])

    # LOGGING
    if cfgs["logs"]["save_local_logs"]:
        log_dir = os.path.join(
            cfgs["logs"]["local_logs_path"],
            cfgs["experiment_name"],
        )
        if not (os.path.exists(log_dir)):
            os.makedirs(log_dir)
        sys.stdout = Logger(os.path.join(log_dir, "{}.log".format(time_stamp)))

    # DATA LOADERS
    transformers = get_transformers(cfgs)
    dataset_train = DeepGlobe_ROM(cfgs, subset="train", transformers=transformers)
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfgs["train_setup"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    dataset_valid = DeepGlobe_ROM(cfgs, subset="val", transformers=transformers)
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=cfgs["train_setup"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # MODEL
    sup_loss = CE_loss
    model = Model(conf=cfgs["model"], sup_loss=sup_loss)
    model = torch.nn.DataParallel(model, device_ids=[0])

    # Training
    metric_fn = eval_metrics
    trainer = Trainer(
        cfgs=cfgs,
        time_stamp=time_stamp,
        model=model,
        device=device,
        loader_train=loader_train,
        loader_valid=loader_valid,
        loss_fn=CE_loss,
        metric_fn=eval_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_file",
        help="config file name/experiment name",
        default="train",
    )
    args = arg_parser.parse_args()
    Cfgs = Config(args)
    main(Cfgs)
