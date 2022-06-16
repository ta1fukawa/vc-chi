# https://github.com/KrishnaDN/x-vector-pytorch

import torch

from modules import global_value as g
from modules import model_parts as mp

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.tdnn1 = mp.TDNN(g.num_mels, 512, 5, 1, dropout_p=0.5)
        self.tdnn2 = mp.TDNN(512, 512, 3, 1, dropout_p=0.5)
        self.tdnn3 = mp.TDNN(512, 512, 2, 2, dropout_p=0.5)
        self.tdnn4 = mp.TDNN(512, 512, 1, 1, dropout_p=0.5)
        self.tdnn5 = mp.TDNN(512, 512, 1, 3, dropout_p=0.5)
        self.segment6 = torch.nn.Linear(1024, 512)
        self.segment7 = torch.nn.Linear(512, 512)
        self.output = torch.nn.Linear(512, g.train_dataset['speaker_end'] - g.train_dataset['speaker_start'])
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        mean = torch.mean(tdnn5_out, dim=1)
        std = torch.std(tdnn5_out, dim=1)
        stat_pooling = torch.cat((mean, std), dim=1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.softmax(self.output(x_vec))
        return predictions, x_vec
