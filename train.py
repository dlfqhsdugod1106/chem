import torch
import argparse
import pytorch_lightning as pl
import os; opj = os.path.join
import shutil
import torch.multiprocessing as multiprocessing
from solver import EnergyPredictionSolver
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, default='chem')
    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--ckpt_metric', type=str, default='')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # log directories
    name = 'debug' if args.debug else datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.save_dir = opj(args.base_dir, name)
    os.makedirs(args.save_dir, exist_ok=True)
    for d in ['valid', 'ckpt', 'wandb']:
        os.makedirs(opj(args.save_dir, d), exist_ok=True)

    # load solver
    solver = EnergyPredictionSolver(args)

    # wandb setup
    use_wandb = not args.debug and args.use_wandb
    logger = WandbLogger(project=args.project, save_dir=opj(args.save_dir, 'wandb')) if use_wandb else None

    # trainer setup
    checkpoint_callback = ModelCheckpoint(dirpath=opj(args.save_dir, 'ckpt'), save_top_k=3, monitor='valid_loss')
    trainer = pl.Trainer(logger=logger,
                         enable_checkpointing=True,
                         max_epochs=10000, 
                         accelerator='gpu',
                         devices=1,
                         fast_dev_run=args.debug,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=args.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback],
                         gradient_clip_val=5)

    if use_wandb and trainer.global_rank == 0:
        logger.experiment.config.update(args)

    trainer.fit(solver)

if __name__ == '__main__':
    train()
