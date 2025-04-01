import sofa
print('Start', flush=True)
from .utils import logging_setup, parse_yaml, initialize_config, ignore_warnings
ignore_warnings()
import traceback
import argparse
import os
import torch
import pathlib
from typing import List, NoReturn
import lightning.pytorch as pl
import json
from pytorch_lightning.utilities.rank_zero import rank_zero_only
print('Finish importing', flush=True)

import logging
logger = logging.getLogger(__name__)

def train(args) -> NoReturn:
    # arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    config_filename = pathlib.Path(config_yaml).stem

    try: tfversion = int(args.version)
    except: tfversion = args.version

    try: tqdm_rate = int(args.tqdm)
    except: tqdm_rate = 0

    log_level = args.log_level.upper() if args.log_level.upper() in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'] else 'INFO'

    tfname = args.tfname

    if args.test:
        workspace='workspace/test'
        tqdm_rate = 1

    # Read config file.
    configs = parse_yaml(config_yaml)

    # deterministic
    if configs['deterministic']:
        torch.use_deterministic_algorithms(True, warn_only=True)
        pl.seed_everything(configs['manual_seed'], workers=True)
        configs['train']['trainer']['args']['deterministic'] = True

    # create directories in workspace
    working_directories = tuple(os.path.join(workspace, config_filename, fol) for fol in ('checkpoints', 'logs', 'tf_logs'))
    for directory in working_directories:
        os.makedirs(directory, exist_ok=True)
    checkpoints_dir, logs_dir, tf_logs_dir = working_directories

    # setup logging
    if rank_zero_only.rank == 0:
        logging_setup(directory=logs_dir, console_log_level=log_level)
        logger.info(args)
        logger.info(json.dumps(configs, indent = 4))

    # data module
    if args.batchsize > 0:
        logger.info(f'Use batchsize of {args.batchsize}')
        configs['datamodule']['args']['train_dataloader']['batch_size'] = args.batchsize
        configs['datamodule']['args']['train_dataloader']['num_workers'] = args.batchsize
        if 'val_dataloader' in configs['datamodule']['args']:
            configs['datamodule']['args']['val_dataloader']['batch_size'] = args.batchsize
        configs['datamodule']['args']['val_dataloader']['num_workers'] = args.batchsize
    
    logger.info('Initialize data module')
    data_module = initialize_config(configs['datamodule'])

    # model
    logger.info('Initialize lightning module')
    pl_model = initialize_config(configs['lightning_module'])

    # callbacks
    logger.info('Initialize callbacks')
    callbacks_configs = configs['train']['callbacks']
    callbacks = []
    for callback_config in callbacks_configs:
        if callback_config['name'] == 'checkpoint': callback_config['args']['dirpath'] = checkpoints_dir
        if callback_config['name'] == 'tqdm' and tqdm_rate > 0:
            callback_config['args']['refresh_rate'] = tqdm_rate
        callback = initialize_config(callback_config)
        callbacks.append(callback)
    
    # tensorboard logger
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tf_logs_dir, name=tfname, version=tfversion)

    # trainer
    configs['train']['trainer']['args']['callbacks'] = callbacks
    configs['train']['trainer']['args']['logger'] = tb_logger
    trainer = initialize_config(configs['train']['trainer'])

    # checkpoints path
    if args.resume_last:
        resume_checkpoint_path = os.path.join(checkpoints_dir, 'last.ckpt')
        logger.info(f'Resume last checkpoint: {resume_checkpoint_path}')
    elif args.resume_checkpoint_path:
        resume_checkpoint_path = args.resume_checkpoint_path
        logger.info(f'Resume training: {args.resume_checkpoint_path}')
    else:
        resume_checkpoint_path = None

    if configs['deterministic']: torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Fit, evaluate, and save checkpoints.
    trainer.fit(
        model=pl_model, 
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=data_module,
        ckpt_path=resume_checkpoint_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        required=False,
        default='workspace/test',
        help="Directory of workspace."
    )

    parser.add_argument(
        "--version", "-v",
        type=str,
        required=False,
        default='',
        help="Version of TensorBoardLogger"
    )

    parser.add_argument(
        "--tfname",
        type=str,
        required=False,
        default='',
        help="filename of TensorBoardLogger"
    )

    parser.add_argument(
        "--tqdm",
        type=str,
        required=False,
        default='0',
        help="refresh rate of tqdm"
    )

    parser.add_argument(
        "--config_yaml", "-c",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser.add_argument(
        "--resume_checkpoint_path", "-r",
        type=str,
        required=False,
        default='',
        help="Path of pretrained checkpoint for finetuning.",
    )

    parser.add_argument(
        "--resume_last",
        action='store_true',
        required=False,
        help="Resume the last.ckpt",
    )

    parser.add_argument(
        "--test",
        action='store_true',
        required=False,
        help="workspace/test, tqdm = 1 "
    )

    parser.add_argument(
        "--log_level", "-l",
        type=str,
        required=False,
        default='INFO',
        help="refresh rate of tqdm"
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        required=False,
        default=-1,
        help="batchsize of train and validation",
    )

    try:
        args = parser.parse_args()
        train(args)
    except:
        traceback.print_exc()
    print('The program is terminated normally')
