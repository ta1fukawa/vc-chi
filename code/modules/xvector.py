# https://github.com/KrishnaDN/x-vector-pytorch

import torch
import torch.nn.functional

from modules import global_value as g
from modules import model_parts as mp


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.first = torch.nn.Sequential(
            torch.nn.Conv2d(1,  64, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(1, 4)),
        
            torch.nn.Conv2d(64,  128, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(1, 4)),
        
            torch.nn.Conv2d(128, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(1, 4)),
        
            torch.nn.Conv2d(256, 2048, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        )

        self.second = torch.nn.Sequential(
            torch.nn.Linear(2048, g.style_dim),
        )

        self.last = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Linear(g.style_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Linear(1024, 16),
            torch.nn.LogSoftmax(dim=-1)
        )

    def _max_pooling(self, x):
        return x.max(dim=3)[0].max(dim=2)[0]

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = self.first(x)
        x = self._max_pooling(x)
        emb = self.second(x)
        x = self.last(emb)

        return x, emb
