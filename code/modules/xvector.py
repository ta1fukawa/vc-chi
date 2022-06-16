# https://github.com/KrishnaDN/x-vector-pytorch

import torch
import torch.nn.functional

from modules import global_value as g
from modules import model_parts as mp

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1a = mp.Layer(1,  64, layer='conv2d', bn=False, activation='relu', kernel_size=(5, 5), padding='same')
        self.conv1b = mp.Layer(64, 64, layer='conv2d', bn=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.pool1  = torch.nn.MaxPool2d(kernel_size=(1, 4))

        self.conv2a = mp.Layer(64,  128, layer='conv2d', bn=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.conv2b = mp.Layer(128, 128, layer='conv2d', bn=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.pool2  = torch.nn.MaxPool2d(kernel_size=(1, 4))

        self.conv3a = mp.Layer(128, 256, layer='conv2d', bn=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.conv3b = mp.Layer(256, 256, layer='conv2d', bn=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.pool3  = torch.nn.MaxPool2d(kernel_size=(1, 4))

        self.conv4  = mp.Layer(256, 2048, layer='conv2d', bn=True, activation='relu', kernel_size=(5, 5), padding='same')
        self.line4  = mp.Layer(2048, g.style_dim, layer='linear', bn=True, activation='linear')

        self.line6a = mp.Layer(g.style_dim, 16, layer='linear', bn=True, activation='linear')
        self.line6b = mp.Layer(g.style_dim, 80, layer='linear', bn=True, activation='linear')

        self.mode = 'small'

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

        if self.mode == 'small':
            x1 = torch.nn.functional.log_softmax(self.line6a(x), dim=-1)
            return x1, emb
        elif self.mode == 'large':
            x2 = torch.nn.functional.log_softmax(self.line6b(x), dim=-1)
            return x2, emb

    def set_train_mode(self, mode):
        self.mode = mode


class NetOld1(torch.nn.Module):
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
