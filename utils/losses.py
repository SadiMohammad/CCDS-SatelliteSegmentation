import torch.nn.functional as F


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    return F.cross_entropy(
        input_logits / temperature, target_targets, ignore_index=ignore_index
    )
