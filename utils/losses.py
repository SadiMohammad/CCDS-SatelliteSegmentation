import torch.nn.functional as F
import torch


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    target_targets = target_targets.type(torch.FloatTensor).cuda(non_blocking=True)
    return F.cross_entropy(input_logits / temperature, target_targets)
