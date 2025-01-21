import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


def hard_negative_mining(output, target, neg_pos_ratio=3):
    """
    Perform hard negative mining to balance the positive and negative samples.
    Args:
        output (torch.Tensor): The model predictions.
        target (torch.Tensor): The ground truth labels.
        neg_pos_ratio (int): The ratio of negative to positive samples to keep.
    Returns:
        torch.Tensor: The indices of the samples to keep.
    """
    pos_mask = target > 0
    neg_mask = target == 0

    pos_loss = F.binary_cross_entropy_with_logits(output[pos_mask], target[pos_mask], reduction='none')
    neg_loss = F.binary_cross_entropy_with_logits(output[neg_mask], target[neg_mask], reduction='none')

    num_pos = pos_mask.sum()
    num_neg = neg_pos_ratio * num_pos

    neg_loss, _ = neg_loss.sort(descending=True)
    neg_loss = neg_loss[:num_neg]

    return torch.cat([pos_loss, neg_loss])


def bce_withlogits_loss_hnm(output, target):
    hard_neg_loss = hard_negative_mining(output, target)
    return hard_neg_loss.mean()
