"""
This script is used to perform cross-validation on GraphSynergy with alternate graphs building methods and their configurations (altering ratios).
It iterates through each possible argument in the graph_ratios list, builds the corresponding graph and performs cross-validation on each fold.

Run the following command from the command line based on the dataset you want to use:
    python cross_validation.py --config ./config/DrugCombDB_cv_config.json
    python cross_validation.py --config ./config/OncologyScreen_cv_config.json
"""

import argparse
import collections
import torch
import numpy as np
import pandas as pd
import data_loader_cross_val.cv_data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.GraphSynergy import GraphSynergy as module_arch
from parse_config import ConfigParser
from trainer.trainer import Trainer

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    """Training."""
    logger = config.get_logger('data loading')
    # specify the graph building method to use as well as the corresponding ratios
    graph_function = config['data_loader']['args']['graph_building_function']
    graph_ratios = config['data_loader']['graph_ratios']

    used_db = config['name']
    test_results = []

    for ratio in graph_ratios:
        logger.info(f"Graph Function: {graph_function}, Graph Ratio: {ratio}")

        # setup data_loader instances
        # Loads and pre-processes the data
        data_loader = module_data.CrossValidationDataLoader(
            data_dir=config['data_loader']['args']['data_dir'],
            batch_size=config['data_loader']['args']['batch_size'],
            score=config['data_loader']['args']['score'],
            n_hop=config['data_loader']['args']['n_hop'],
            n_memory=config['data_loader']['args']['n_memory'],
            num_folds=config['data_loader']['args']['num_folds'],
            shuffle=config['data_loader']['args']['shuffle'],
            num_workers=config['data_loader']['args']['num_workers'],
            graph_building_function=graph_function,
            graph_ratio=ratio
        )
        # get functions extract specific data attributes for use in the model
        feature_index = data_loader.get_feature_index()
        cell_neighbor_set = data_loader.get_cell_neighbor_set()
        drug_neighbor_set = data_loader.get_drug_neighbor_set()
        node_num_dict = data_loader.get_node_num_dict()

        # get folds from the data_loader and iterate through each
        for fold_id in data_loader.get_fold_indices().keys():
            # setup config and logger for new fold
            config.fold_id = fold_id
            logger = config.get_logger(f'train-f{fold_id}')

            # (an instance of GraphSynergy) is initialized with parameters like protein_num and cell_num
            model = module_arch(protein_num=node_num_dict['protein'],
                                cell_num=node_num_dict['cell'],
                                drug_num=node_num_dict['drug'],
                                emb_dim=config['arch']['args']['emb_dim'],
                                n_hop=config['arch']['args']['n_hop'],
                                l1_decay=config['arch']['args']['l1_decay'],
                                therapy_method=config['arch']['args']['therapy_method'])
            logger.info(model)

            # get function handles of loss and metrics
            # Specifies the loss function (error function) from module_loss
            criterion = getattr(module_loss, config['loss'])
            # List of functions (like accuracy or precision) to evaluate model performance.
            metrics = [getattr(module_metric, met) for met in config['metrics']]

            # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for disabling scheduler
            # List of functions (like accuracy or precision) to evaluate model performance
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            # Manages gradient updates
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

            lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
            # assign row indices of folds to be training, validation and test indices (as described in cv_base_data_loader)
            data_loader.set_folds(fold_id)
            # assign the fold indices accordingly to be training, validation and test sets (as described in cv_base_data_loader)
            train_loader = data_loader.get_train_loader()
            val_loader = data_loader.get_val_loader()
            test_loader = data_loader.get_test_loader()

            print(f"Fold {fold_id}:")
            print(f"  Training set: {len(train_loader.dataset)} samples")
            print(f"  Validation set: {len(val_loader.dataset)} samples")
            print(f"  Testing set: {len(test_loader.dataset)} samples")

            # Initialize the training with the matching sets as arguments and begin training
            trainer = Trainer(model, criterion, metrics, optimizer,
                              config=config,
                              data_loader=train_loader,
                              feature_index=feature_index,
                              cell_neighbor_set=cell_neighbor_set,
                              drug_neighbor_set=drug_neighbor_set,
                              valid_data_loader=val_loader,
                              test_data_loader=test_loader,
                              lr_scheduler=lr_scheduler)
            trainer.train()

            """Testing."""
            logger = config.get_logger(f'test-fold-f{fold_id}')
            logger.info(model)
            test_metrics = [getattr(module_metric, met) for met in config['metrics']]

            # load best checkpoint
            resume = str(config.save_dir / f'model_best_fold_{fold_id}.pth')
            logger.info('Loading checkpoint: {} ...'.format(resume))
            checkpoint = torch.load(resume, weights_only=False)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)

            test_output = trainer.test()
            log = {'loss': test_output['total_loss'] / test_output['n_samples']}
            log.update({
                met.__name__: test_output['total_metrics'][i].item() / test_output['n_samples'] \
                for i, met in enumerate(test_metrics)
            })
            logger.info(log)

            results = {'fold_id': fold_id, 'graph_function': graph_function, 'graph_ratio': ratio}
            results.update(log)
            test_results.append(pd.Series(results))
        # Exit successfully
        print("Cross validation completed successfully:")

    test_results = pd.DataFrame(test_results)
    test_results.to_csv(config.log_dir / f'{graph_function}_{used_db}_test_results_.csv', index=False)
    print(test_results)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
