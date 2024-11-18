import torch
from sklearn.model_selection import KFold
import torch.utils.data as data

def cross_validation(data_loader, n_splits=5):
    """
    Perform K-Fold Cross-Validation on the given dataset.
    """
    dataset = data_loader.dataset
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(dataset):
        train_subset = data.Subset(dataset, train_idx)
        val_subset = data.Subset(dataset, val_idx)
        yield train_subset, val_subset
