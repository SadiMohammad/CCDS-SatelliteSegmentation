import os
import torch
from tqdm import tqdm
from utils.metrics import eval_metrics


class Trainer:
    def __init__(
        self, **kwargs
    ):  # time_stamp, model, optimizer, device, loader_train, loader_valid, loss_fn, metric_fn
        self.__dict__.update(kwargs)

    def train(self):
        self.optimizer.zero_grad()
        best_valid_score = self.cfgs["train_setup"]["best_valid_score"]
        for epoch in range(self.cfgs["train_setup"]["epochs"]):
            print("\tStarting epoch - {}/{}.".format(epoch + 1, self.epochs))
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
                    total_loss, cur_losses, outputs = self.model(
                        x_l=input, target_l=target
                    )

                if self.cfgs["train_setup"]["use_thld_for_valid"]:
                    outputs = (
                        outputs > self.cfgs["train_setup"]["thld_for_valid"]
                    ).float()
                correct, labeled, inter, union = eval_metrics(
                    outputs, target, self.num_classes, self.ignore_index
                )
            print(inter)
            print(union)
            # print("\n")
            # print(
            #     "VALIDATION >>> epoch: {:04d}/{:04d}, running_metric: {}".format(
            #         epoch + 1,
            #         self.epochs,
            #         epoch_valid_metric,
            #     ),
            #     end="\r",
            # )
            # print("\n" * 2)
        return epoch_valid_metric
