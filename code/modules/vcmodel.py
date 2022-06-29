import torch

from modules import global_value as g
from modules import model_parts as mp


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.content_enc = ContentEncoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, c: torch.Tensor, c_emb: torch.Tensor, s_emb: torch.Tensor):
        feat  = self.content_enc(c, c_emb)
        r     = self.decoder(feat, s_emb)
        q     = r + self.postnet(r)
        return q


class ContentEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = mp.Layer(g.num_mels + g.style_dim, 512, layer='conv1d', bn=True, bn_first=False, kernel_size=5, padding='same', activation='relu')
        self.conv2 = mp.Layer(512, 512, layer='conv1d', bn=True, bn_first=False, kernel_size=5, padding='same', activation='relu')
        self.conv3 = mp.Layer(512, 512, layer='conv1d', bn=True, bn_first=False, kernel_size=5, padding='same', activation='relu')

        self.lstm = torch.nn.LSTM(512, g.dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, c: torch.Tensor, x_emb: torch.Tensor):
        x_emb = x_emb.unsqueeze(1).expand(-1, c.size(1), -1)  # Expand to time dimension
        x = torch.cat([c, x_emb], dim=-1)  # Concatenate in frequency dimension

        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        code = torch.cat([
            x[:, g.lstm_stride - 1::g.lstm_stride, :g.dim_neck],
            x[:, :-g.lstm_stride + 1:g.lstm_stride, g.dim_neck:]
        ], dim=-1)
        x = code.repeat_interleave(c.size(1) // code.size(1), dim=1)

        return x


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.lstm1 = torch.nn.LSTM(g.dim_neck * 2 + g.style_dim, 512, 1, batch_first=True)
        self.conv1 = mp.Layer(g.dim_neck * 2 + g.style_dim, 512, layer='conv1d', bn=True, bn_first=False, kernel_size=5, padding='same', activation='relu')
        self.conv2 = mp.Layer(512, 512, layer='conv1d', bn=True, bn_first=False, kernel_size=5, padding='same', activation='relu')
        self.conv3 = mp.Layer(512, 512, layer='conv1d', bn=True, bn_first=False, kernel_size=5, padding='same', activation='relu')
        self.lstm2 = torch.nn.LSTM(512, 1024, 3, batch_first=True)
        self.line = mp.Layer(1024, g.num_mels, layer='linear')

    def forward(self, feat: torch.Tensor, s_emb: torch.Tensor):
        s_emb = s_emb.unsqueeze(1).expand(-1, feat.size(1), -1)  # Expand to time dimension
        x = torch.cat([feat, s_emb], dim=-1)  # Concatenate in frequency dimension

        # x, _ = self.lstm1(x)

        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm2(x)
        x = self.line(x)

        return x


class Postnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = mp.Layer(80,  512, layer='conv1d', bn=True, activation='tanh',   kernel_size=5, padding='same')
        self.conv2 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='tanh',   kernel_size=5, padding='same')
        self.conv3 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='tanh',   kernel_size=5, padding='same')
        self.conv4 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='tanh',   kernel_size=5, padding='same')
        self.conv5 = mp.Layer(512, 80,  layer='conv1d', bn=True, activation='linear', kernel_size=5, padding='same')

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.transpose(1, 2)

        return x