import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class CrossValidationBaseDataLoader:
    """
    Base class for cross validation data loader
    """
    def __init__(self, dataset, batch_size, folds_dict, shuffle, num_workers,
                 collate_fn=default_collate):
        self.folds = folds_dict
        self.num_folds = len(folds_dict.keys())
        self.dataset = dataset

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # Initialize remaining arguments for DataLoader
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

    # assign folds to be training, validation and test folds based on a given fold id
    def set_folds(self, fold_id):
        # fold with given id will be used for validation
        val_fold = fold_id
        # the next fold (or the first fold if val_fold is the last fold) will be used for testing
        test_fold = (fold_id + 1) % self.num_folds
        # assign remaining folds to be used for training the model
        train_folds = [idx for idx in range(self.num_folds) if idx != val_fold and idx != test_fold]

        # assign the row indices of the folds according to their role
        self.val_indices = self.folds[val_fold]
        self.test_indices = self.folds[test_fold]
        self.train_indices = [idx for fold in train_folds for idx in self.folds[fold]]

    # ChatGPT Template
    def get_train_loader(self):
        """
        Get the DataLoader for the training set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the training set.
        """
        # sample the rows of the training folds from the dataset
        train_subset = Subset(self.dataset, self.train_indices)
        return DataLoader(
            train_subset,
            **self.init_kwargs
        )

    # ChatGPT Template
    def get_val_loader(self):
        """
        Get the DataLoader for the validation set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation set.
        """
        # sample the rows of the validation fold from the dataset
        val_subset = Subset(self.dataset, self.val_indices)
        return DataLoader(
            val_subset,
            **self.init_kwargs
        )

    # ChatGPT Template
    def get_test_loader(self):
        """
        Get the DataLoader for the testing set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the testing set.
        """
        # sample the rows of the test folds from the dataset
        test_subset = Subset(self.dataset, self.test_indices)
        return DataLoader(
            test_subset,
            **self.init_kwargs
        )
