# https://github.com/KrishnaDN/x-vector-pytorch

import logging
import torch
import torch.nn.functional

from modules import global_value as g
from modules import model_parts as mp


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.tdnn = torch.nn.Sequential(
            mp.TDNN(g.num_mels, 512, context_size=5, dilation=1, dropout_p=0.5),
            mp.TDNN(512, 512, context_size=3, dilation=1, dropout_p=0.5),
            mp.TDNN(512, 512, context_size=2, dilation=2, dropout_p=0.5),
            mp.TDNN(512, 512, context_size=1, dilation=1, dropout_p=0.5),
            mp.TDNN(512, 512, context_size=1, dilation=3, dropout_p=0.5),
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.Linear(512, 512),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, g.speaker_size),
            torch.nn.LogSoftmax(dim=1),
        )

    def set_cassifier(self, speaker_size):
        if speaker_size == self.classifier[0].out_features:
            return

        self.classifier[0] = torch.nn.Linear(self.classifier[0].in_features, speaker_size).to(g.device)
        logging.info('Set classifier size to {}'.format(speaker_size))

    def _stats_pooling(self, x):
        mean = torch.mean(x, dim=1)
        std  = torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)

    def forward(self, x):
        x = self.tdnn(x)

        mean = torch.mean(x, dim=1)
        std  = torch.std(x, dim=1)
        stats_pooling = torch.cat((mean, std), dim=1)

        emb = self.linear(stats_pooling)
        x = self.classifier(emb)

        return x, emb
