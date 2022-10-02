import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import os; opj = os.path.join
import numpy as np
import matplotlib.pyplot as plt 

from dataset import LiBF4Dataset
from model import FFAwareMLPEnergyPredictor
from plot import plot_prediction

class EnergyPredictionSolver(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.__dict__.update(**vars(args))
        self.save_hyperparameters()
        self.get_dataset()
        self.model = FFAwareMLPEnergyPredictor()

    def step(self, batch, idx, prefix):
        pred = self.model(batch['parameter'], batch['energy_ff'])
        loss = F.l1_loss(pred, batch['energy_residual'])
        self.log_dict({f'{prefix}_loss': loss})
        return loss, pred

    def training_step(self, batch, idx):
        loss, pred = self.step(batch, idx, 'train')
        return loss

    def validation_step(self, batch, idx):
        loss, pred = self.step(batch, idx, 'valid')
        self.energy_ff.append(batch['energy_ff'].detach().cpu().numpy())
        self.energy_true.append(batch['energy_sapt'].detach().cpu().numpy())
        self.pred_residual.append(pred.detach().cpu().numpy())
        return loss

    def predict_step(self, batch, idx):
        pred = self.model(batch['parameter'], batch['energy_ff'])
        pred = pred+batch['energy_ff']
        return pred

    def on_validation_epoch_start(self):
        self.energy_ff, self.pred_residual, self.energy_true = [], [], []

    def on_validation_epoch_end(self):
        self.energy_ff, self.pred_residual, self.energy_true = map(np.concatenate, (self.energy_ff, self.pred_residual, self.energy_true))
        fig = plot_prediction(self.energy_ff, self.pred_residual, self.energy_true)
        fig.savefig(opj(self.save_dir, 'valid', f'{str(self.current_epoch).zfill(3)}.pdf'), bbox_inches='tight')
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.parameters(), lr=self.lr, eps=1e-4, weight_decay=self.weight_decay)

    def log_dict(self, log_dict):
        super().log_dict(log_dict, prog_bar=True, logger=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

    def get_dataset(self):
        full_dataset = LiBF4Dataset(data_dir=self.data_dir)
        num_train_data = int(0.9*len(full_dataset))
        num_valid_data = len(full_dataset)-num_train_data
        self.train_set, self.valid_set = torch.utils.data.random_split(full_dataset, [num_train_data, num_valid_data])

    def get_dataloader(self, split):
        dataset = self.train_set if split == 'train' else self.valid_set
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True if split == 'train' else False, 
                          num_workers=4, persistent_workers=True, pin_memory=True)
                                                   
    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('valid')
