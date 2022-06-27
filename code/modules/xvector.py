# https://github.com/KrishnaDN/x-vector-pytorch

import logging
import torch
import torch.nn.functional

from modules import global_value as g
from modules import model_parts as mp


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1,  64, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 4)),

            torch.nn.Conv2d(64,  128, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 4)),

            torch.nn.Conv2d(128, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=(2, 4)),

            torch.nn.Conv2d(256, 2048, kernel_size=(5, 5), dilation=(1, 1), padding='same'),
        )

        self.compress = torch.nn.Sequential(
            torch.nn.Linear(2048, g.style_dim),
        )

        self.cushion = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),

            # torch.nn.Linear(g.style_dim, 1024),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(g.style_dim, g.speaker_size),
            torch.nn.LogSoftmax(dim=-1),
        )

    def set_cassifier(self, speaker_size):
        if speaker_size == self.classifier[0].out_features:
            return

        self.classifier[0] = torch.nn.Linear(self.classifier[0].in_features, speaker_size).to(g.device)
        logging.info('Set classifier size to {}'.format(speaker_size))

    def _max_pooling(self, x):
        return x.max(dim=3)[0].max(dim=2)[0]

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = self.conv(x)
        x = self._max_pooling(x)
        emb = self.compress(x)
        x = self.cushion(emb)
        x = self.classifier(x)

        return x, emb
