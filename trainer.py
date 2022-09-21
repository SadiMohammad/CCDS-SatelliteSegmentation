import os
import torch
from tqdm import tqdm
from base import BaseTrainer
import numpy as np


class Trainer:
    def __init__(
        self, **kwargs
    ):  # cfgs, time_stamp, model, optimizer, device, loader_train, loader_valid, loss_fn, metric_fn
        self.__dict__.update(kwargs)
        self.mode = self.cfgs["model"]["mode"]
        self.num_classes = self.cfgs["model"]["num_classes"]

    def train(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.optimizer.zero_grad()
        best_valid_score = self.cfgs["train_setup"]["best_valid_score"]
        for epoch in range(self.cfgs["train_setup"]["epochs"]):
            print(
                "\tStarting epoch - {}/{}.".format(
                    epoch + 1, self.cfgs["train_setup"]["epochs"]
                )
            )
            self.model.train()
            total_batch_loss = 0

            for i_train, sample_train in enumerate(tqdm(self.loader_train)):
                input = sample_train["images"]
                target = sample_train["labels"]
                input, target = input.cuda(non_blocking=True), target.cuda(
                    non_blocking=True
                )

                if self.mode == "sup":
                    total_loss, cur_losses, outputs = self.model(
                        x_l=input, target_l=target
                    )

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_validation_metric = self._validate(epoch)
            if epoch_validation_metric > best_valid_score:
                best_valid_score = epoch_validation_metric
                if self.cfgs["train_setup"]["save_best_model"]:
                    save_ckpts = {
                        "epoch": epoch,
                        "input_size": self.input_size,
                        "best_score": best_valid_score,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "model_state_dict": self.model.state_dict(),
                    }
                    torch.save(
                        save_ckpts,
                        os.path.join(
                            self.cfgs["train_setup"]["checkpoints_path"],
                            self.cfgs["model"]["model_name"],
                            "{}.pth".format(self.time_stamp),
                        ),
                    )
                    print("!!! Checkpoint {} saved !!!".format(epoch + 1))

    def _validate(self, epoch):
        self.model.eval()
        epoch_valid_metric = 0
        with torch.no_grad():
            for i_valid, sample_valid in enumerate(self.loader_valid):
                input = sample_valid["images"]
                target = sample_valid["labels"]
                input, target = input.cuda(non_blocking=True), target.cuda(
                    non_blocking=True
                )
                if self.mode == "sup":
                    outputs = self.model(x_l=input, target_l=target)

                if self.cfgs["train_setup"]["use_thld_for_valid"]:
                    outputs = (
                        outputs > self.cfgs["train_setup"]["thld_for_valid"]
                    ).float()
                inter, union = self.metric_fn(outputs, target, self.num_classes)

            epoch_valid_metric = np.mean(np.nan_to_num(inter / union))
            print("\n")
            print(
                "VALIDATION >>> epoch: {:04d}/{:04d}, running_metric: {}".format(
                    epoch + 1,
                    self.cfgs["train_setup"]["epochs"],
                    epoch_valid_metric,
                ),
                end="\r",
            )
            print("\n" * 2)
        return epoch_valid_metric

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if "class_iou" not in k:
                self.writer.add_scalar(f"train/{k}", v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f"train/Learning_rate_{i}", opt_group["lr"], self.wrt_step
            )
        # current_rampup = self.model.module.unsup_loss_w.current_rampup
        # self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)
