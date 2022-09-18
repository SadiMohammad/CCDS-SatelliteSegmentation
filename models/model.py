import torch
import torch.nn.functional as F
from torch import nn
from models.encoder import Encoder
from models.modeling.deeplab import DeepLab as DeepLab_v3p
import numpy as np


class Model:
    def __init__(
        self,
        conf,
        sup_loss=None,
        ignore_index=None,
        testing=False,
        pretrained=True,
    ):
        self.conf = conf
        self.mode = conf["mode"]

        self.ignore_index = ignore_index

        self.num_classes = conf["num_classes"]
        self.sup_loss_w = conf["supervised_w"]
        self.sup_loss = sup_loss
        self.backbone = conf["backbone"]
        self.layers = conf["layers"]
        self.out_dim = conf["out_dim"]

        assert self.layers in [50, 101]

        if self.backbone == "deeplab_v3+":
            self.encoder = DeepLab_v3p(backbone="resnet{}".format(self.layers))
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1),
            )
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == "psp":
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(
                self.out_dim, self.num_classes, kernel_size=1, stride=1
            )
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

    def forward(self, x_l=None, target_l=None):
        if not self.training:
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            return F.interpolate(
                enc, size=x_l.size()[2:], mode="bilinear", align_corners=True
            )

        if self.mode == "supervised":
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            output_l = F.interpolate(
                enc, size=x_l.size()[2:], mode="bilinear", align_corners=True
            )

            loss_sup = (
                self.sup_loss(
                    output_l, target_l, ignore_index=self.ignore_index, temperature=1.0
                )
                * self.sup_loss_w
            )

            curr_losses = {"loss_sup": loss_sup}
            outputs = {"sup_pred": output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs
