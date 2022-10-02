import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
from einops import repeat

class MLPEnergyPredictor(nn.Module):
    def __init__(self,):
        super().__init__() 
        self.net = nn.Sequential(nn.Linear(3, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 1, bias=False))

    def forward(self, parameter):
        return self.net(parameter).sum((-1, -2))

class FFAwareMLPEnergyPredictor(nn.Module):
    def __init__(self,):
        super().__init__() 
        self.net = nn.Sequential(nn.Linear(4, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 1, bias=False))

    def forward(self, parameter, energy_ff):
        # approx. normalization
        energy_ff = (550+energy_ff)/100 
        energy_ff = repeat(energy_ff, 'b -> b n 1', n=6)
        x = torch.cat([parameter, energy_ff], -1)
        return self.net(x).sum((-1, -2))
