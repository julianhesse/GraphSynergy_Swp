import argparse
import collections
import torch
import numpy as np
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
    logger = config.get_logger('train')

    # setup data_loader instances
    # Loads and pre-processes the data
    data_loader = config.init_obj('data_loader', module_data)
    # creates validation and test datasets
    # get functions extract specific data attributes for use in the model
    feature_index = data_loader.get_feature_index()
    cell_neighbor_set = data_loader.get_cell_neighbor_set()
    drug_neighbor_set = data_loader.get_drug_neighbor_set()
    node_num_dict = data_loader.get_node_num_dict()

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

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # List of functions (like accuracy or precision) to evaluate model performance
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Manages gradient updates
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    for fold_id in data_loader.get_fold_indices().keys():
        data_loader.set_folds(fold_id)

        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        test_loader = data_loader.get_test_loader()

        print(f"Fold {fold_id}:")
        print(f"  Training set: {len(train_loader.dataset)} samples")
        print(f"  Validation set: {len(val_loader.dataset)} samples")
        print(f"  Testing set: {len(test_loader.dataset)} samples")

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
        logger = config.get_logger('test')
        logger.info(model)
        test_metrics = [getattr(module_metric, met) for met in config['metrics']]

        # load best checkpoint
        resume = str(config.save_dir / 'model_best.pth')
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

    print("Cross validation completed successfully:")


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