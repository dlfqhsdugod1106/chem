import numpy as np
from torch.utils.data import Dataset 
import os; opj = os.path.join
from einops import rearrange

class LiBF4Dataset(Dataset):
    def __init__(self, data_dir='data'):
        super().__init__()
        self.parameter_data, self.energy_data = self.construct_dataset(data_dir)

    def construct_dataset(self, data_dir):
        parameter_data = np.load(opj(data_dir, 'tot_param.npy'))
        parameter_data = rearrange(parameter_data, 'b (d n) -> b n d', n=6)
        energy_data = np.load(opj(data_dir, 'tot_ene.npy'))
        parameter_data, energy_data = parameter_data.astype(np.float32), energy_data.astype(np.float32)
        return parameter_data, energy_data

    def __getitem__(self, idx):
        parameter, energy = self.parameter_data[idx], self.energy_data[idx]
        return dict(parameter=parameter, 
                    energy_sapt=energy[0], energy_ff=energy[1], energy_residual=energy[2])

    def __len__(self):
        return self.parameter_data.shape[0]

class LiBF4InferenceDataset(Dataset):
    def __init__(self, param_dir='data/tot_param.npy', ff_energy_dir='data/ff_ene.npy'):
        super().__init__()
        self.parameter_data, self.energy_data = self.construct_dataset(param_dir, ff_energy_dir)

    def construct_dataset(self, param_dir, ff_energy_dir):
        parameter_data = np.load(param_dir)
        parameter_data = rearrange(parameter_data, 'b (d n) -> b n d', n=6)
        energy_data = np.load(ff_energy_dir)
        parameter_data, energy_data = parameter_data.astype(np.float32), energy_data.astype(np.float32)
        return parameter_data, energy_data

    def __getitem__(self, idx):
        parameter = self.parameter_data[idx]
        parameter, energy = self.parameter_data[idx], self.energy_data[idx]
        return dict(parameter=parameter, energy_ff=energy)

    def __len__(self):
        return self.parameter_data.shape[0]
