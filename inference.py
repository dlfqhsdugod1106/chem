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
from dataset import LiBF4InferenceDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

def inference():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, default='ckpt/epoch\=1819-step\=220220.ckpt')
    parser.add_argument('--param_dir', type=str, default='data/tot_param.npy')
    parser.add_argument('--ff_energy_dir', type=str, default='data/ff_ene.npy')
    parser.add_argument('--save_dir', type=str, default='data/inference_results.npy')
    parser.add_argument('--batch_size', type=int, default=128)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    solver = EnergyPredictionSolver.load_from_checkpoint(args.ckpt) 
    trainer = pl.Trainer(accelerator='gpu', devices=-1, max_epochs=-1, num_sanity_val_steps=0)

    inference_dataset = LiBF4InferenceDataset(args.param_dir, args.ff_energy_dir)
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    predictions = trainer.predict(solver, dataloaders=inference_loader)
    predictions = torch.cat(predictions).detach().cpu().numpy()
    np.save(args.save_dir, predictions)

if __name__ == '__main__':
    inference()
