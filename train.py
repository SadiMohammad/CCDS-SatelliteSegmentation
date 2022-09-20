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
from utils.losses import CE_loss
from dataloaders.deepglobe import DeepGlobe_ROM
from trainer import Trainer

time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")


def main(Cfgs):
    cfgs = Cfgs.cfgs
    device = torch.device(cfgs["train_setup"]["device"])

    # LOGGING
    if cfgs["logs"]["save_local_logs"]:
        sys.stdout = Logger(
            os.path.join(
                cfgs["logs"]["local_logs_path"],
                cfgs["experiment_name"],
                "{}.log".format(time_stamp),
            )
        )

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
    model = Model(
        num_classes=cfgs["model"]["num_classes"],
        conf=cfgs["model"],
        sup_loss=sup_loss,
    )

    # Training
    optimizer = getattr(optim, cfgs["optimizer"]["optimizer_fn"])(
        model.parameters(),  # momentum=0.9, for sgd
        lr=cfgs["optimizer"]["initial_lr"],
        weight_decay=0.0005,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfgs["optimizer"]["optimizer_fn"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )
    metric_fn = cfgs["train_setup"]["metric"]
    trainer = Trainer(
        time_stamp=time_stamp,
        model=model,
        optimizer=optimizer,
        device=device,
        loader_train=loader_train,
        loader_valid=loader_valid,
        loss_fn=CE_loss,
        metric_fn=metric_fn,
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
