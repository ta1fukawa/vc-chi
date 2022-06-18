# https://github.com/KrishnaDN/x-vector-pytorch

import torch
import torch.nn.functional

from modules import global_value as g
from modules import model_parts as mp


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1a = mp.Layer(1,  64, layer='conv2d', bn=False, bn_first=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.conv1b = mp.Layer(64, 64, layer='conv2d', bn=True,  bn_first=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.pool1  = torch.nn.MaxPool2d(kernel_size=(1, 4))

        self.conv2a = mp.Layer(64,  128, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.conv2b = mp.Layer(128, 128, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.pool2  = torch.nn.MaxPool2d(kernel_size=(1, 4))

        self.conv3a = mp.Layer(128, 256, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.conv3b = mp.Layer(256, 256, layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.pool3  = torch.nn.MaxPool2d(kernel_size=(1, 4))

        self.conv4  = mp.Layer(256,  2048,        layer='conv2d', bn=True, bn_first=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.line4  = mp.Layer(2048, g.style_dim, layer='linear', bn=True, bn_first=True, activation='linear')

        self.line6 = mp.Layer(g.style_dim, 80, layer='linear', bn=True, bn_first=True, activation='linear')

    def _max_pooling(self, x):
        return x.max(dim=3)[0].max(dim=2)[0]

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self._max_pooling(x)
        emb = self.line4(x)

        x = torch.nn.functional.relu(emb)

        x = torch.nn.functional.log_softmax(self.line6(x), dim=-1)

        return x, emb
