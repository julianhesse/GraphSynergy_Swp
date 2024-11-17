import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class CrossValidationBaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, folds_dict, shuffle, num_workers,
                 collate_fn=default_collate):
        self.folds = folds_dict
        self.n_samples = len(dataset)
        self.num_folds = len(folds_dict.keys())
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

    def set_folds(self, fold_id):

        val_fold = fold_id
        test_fold = (fold_id + 1) % self.num_folds
        train_folds = [idx for idx in range(self.num_folds) if idx != val_fold and idx != test_fold]

        self.val_indices = self.folds[val_fold]
        self.test_indices = self.folds[test_fold]
        self.train_indices = [idx for fold in train_folds for idx in self.folds[fold]]

    # ChatGPT
    def get_train_loader(self):
        """
        Get the DataLoader for the training set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the training set.
        """
        train_subset = Subset(self.dataset, self.train_indices)
        return DataLoader(
            train_subset,
            **self.init_kwargs
        )

    # ChatGPT
    def get_val_loader(self):
        """
        Get the DataLoader for the validation set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation set.
        """
        val_subset = Subset(self.dataset, self.val_indices)
        return DataLoader(
            val_subset,
            **self.init_kwargs
        )

    # ChatGPT
    def get_test_loader(self):
        """
        Get the DataLoader for the testing set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the testing set.
        """
        test_subset = Subset(self.dataset, self.test_indices)
        return DataLoader(
            test_subset,
            **self.init_kwargs
        )
