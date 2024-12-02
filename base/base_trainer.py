import torch
import pandas as pd
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

import wandb


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_fns, optimizer, config):
        self.config = config
        if config.has_fold:
            self.logger = config.get_logger(f'trainer-f{self.config.fold_id}', config['trainer']['verbosity'])
        else:
            self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_fns = metric_fns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            # mnt_mode determines minimum or maximum metric is best, if we use loss, thus mnt_mode = min
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        # setup metric logger
        self.log_metrics = cfg_trainer.get('log_metrics', False)
        self.log_dir = config.log_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        use_wandb = wandb.run is not None
        not_improved_count = 0
        metrics = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # Conditionally log to WandB
            if use_wandb:
                # Flatten train and validation metrics for better visualization
                wandb_log_data = {'epoch': epoch}
                train_metrics = log.get('train', {})
                val_metrics = log.get('validation', {})

                # Add train and validation metrics to WandB log data
                wandb_log_data.update({f"train_{k}": v for k, v in train_metrics.items()})

                # Add validation metrics with 'val_' prefix, avoiding redundant prefixes
                wandb_log_data.update({
                    k if k.startswith("val_") else f"val_{k}": v
                    for k, v in val_metrics.items()
                })

                # Calculate and log metric gaps between training and validation
                for metric_name in train_metrics.keys():
                    val_metric_name = f"val_{metric_name}"
                    if val_metric_name in val_metrics:  # Ensure the metric exists in both train and validation
                        gap = train_metrics[metric_name] - val_metrics[val_metric_name]
                        wandb_log_data[f"gap_{metric_name}"] = gap

                # Log all data to WandB
                wandb.log(wandb_log_data)

            # print logged informations to the screen
            self.logger.info('epoch: {}'.format(epoch))
            for key in ['train', 'validation']:
                if key not in log:
                    continue
                value_format = ''.join(['{:15s}: {:.2f}\t'.format(k, v) for k, v in log[key].items()])
                self.logger.info('    {:15s}: {}'.format(str(key), value_format))

            # collect metrics for later analysis
            if self.log_metrics:
                epoch_metrics = {'epoch': int(epoch)}
                epoch_metrics.update(log.get('train', {}))
                epoch_metrics.update(log.get('validation', {}))
                metrics.append(pd.Series(epoch_metrics))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self._save_checkpoint(epoch, save_best=best)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

        # save the metrics
        if self.log_metrics:
            metrics = pd.DataFrame(metrics)
            if self.config.has_fold:
                metrics.to_csv(self.log_dir / f'metrics_fold_{self.config.fold_id}.csv', index=False)
            else:
                metrics.to_csv(self.log_dir / 'metrics.csv', index=False)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if torch.backends.mps.is_available():
            self.logger.info("Apple Device detected: using Metal Performance Shaders (MPS)")
            device = torch.device('mps' if n_gpu_use > 0 else 'cpu')
        elif n_gpu_use > 0 and n_gpu > 0:
            self.logger.info("CUDA Device detected: using NVIDIA GPU")
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            if self.config.has_fold:
                best_path = str(self.checkpoint_dir / f'model_best_fold_{self.config.fold_id}.pth')
                torch.save(state, best_path)
                self.logger.info(f"Saving current best: model_best_fold_{self.config.fold_id}.pth ...")
            else:
                best_path = str(self.checkpoint_dir / 'model_best.pth')
                torch.save(state, best_path)
                self.logger.info("Saving current best: model_best.pth ...")
        else:
            if self.config.has_fold:
                filename = str(self.checkpoint_dir / 'checkpoint-epoch{}_fold{}.pth'.format(epoch, self.config.fold_id))
            else:
                filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
