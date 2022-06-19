# https://github.com/KrishnaDN/x-vector-pytorch

import torch
import torch.nn.functional

from modules import global_value as g
from modules import model_parts as mp


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.first = torch.nn.Sequential(
            mp.Layer(1, 512, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(3, 7), padding='same'),
            torch.nn.MaxPool2d(kernel_size=(2, 4)),
            mp.Layer(512, 512, layer='conv2d', bn=True,  bn_first=True, activation='relu', kernel_size=(3, 5), padding='same'),
            torch.nn.MaxPool2d(kernel_size=(2, 4)),
            mp.Layer(512, 1024, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(3, 5), padding='same'),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            mp.Layer(1024, 1024, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(3, 5), padding='same'),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            mp.Layer(1024, 2048, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(3, 3), padding='same'),
            mp.Layer(2048, 2048, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(1, 1), padding='same'),
        )

        self.second = torch.nn.Sequential(
            mp.Layer(2048, 1024, layer='linear', bn=True, bn_first=True, activation='linear'),
            mp.Layer(1024, g.style_dim, layer='linear', bn=True, bn_first=True, activation='linear'),
        )

        self.last = torch.nn.Sequential(
            torch.nn.ReLU(),
            mp.Layer(g.style_dim, 80, layer='linear', bn=True, bn_first=True, activation='linear'),
            torch.nn.LogSoftmax(dim=-1),
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
