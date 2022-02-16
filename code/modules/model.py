import torch

from modules import global_value as g
from modules import model_parts as mp

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.content_enc = ContentEncoder()
        self.style_enc = StyleEncoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, c: torch.Tensor, s: torch.Tensor):
        c_emb = self.style_enc(c)
        s_emb = self.style_enc(s)
        code  = self.content_enc(c, c_emb)
        r     = self.decoder(code, s_emb)
        q     = r + self.postnet(r)
        return q

class ContentEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = mp.Layer(g.nmels + g.style_dim, 512, layer='conv1d', bn=True, activation='relu', kernel_size=5, padding='same')
        self.conv2 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='relu', kernel_size=5, padding='same')
        self.conv3 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='relu', kernel_size=5, padding='same')

        self.lstm = torch.nn.LSTM(512, g.dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, c: torch.Tensor, x_emb):
        x_emb = x_emb.unsqueeze(1).expand(-1, c.size(1), -1)  # Expand to time dimension
        x = torch.cat([c, x_emb], dim=-1)  # Concatenate in frequency dimension

        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = torch.cat([
            x[:, g.lstm_stride - 1::g.lstm_stride, :g.dim_neck],
            x[:, :-g.lstm_stride + 1:g.lstm_stride, g.dim_neck:]
        ], dim=-1)
        x = x.repeat_interleave(c.size(1) // x.size(1), dim=1)

        return x

class StyleEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(g.nmels, 768, 3, batch_first=True)
        self.line = mp.Layer(768, g.style_dim, layer='linear')

    def forward(self, x: torch.Tensor):
        _, (hn, _) = self.lstm(x)
        x = self.line(hn[-1])

        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(g.dim_neck * 2 + g.style_dim, 512, 1, batch_first=True)
        self.conv1 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='relu', kernel_size=5, padding='same')
        self.conv2 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='relu', kernel_size=5, padding='same')
        self.conv3 = mp.Layer(512, 512, layer='conv1d', bn=True, activation='relu', kernel_size=5, padding='same')
        self.lstm2 = torch.nn.LSTM(512, 1024, 2, batch_first=True)
        self.line = mp.Layer(1024, g.nmels, layer='linear', activation='linear')

    def forward(self, code: torch.Tensor, s_emb: torch.Tensor):
        s_emb = s_emb.unsqueeze(1).expand(-1, code.size(1), -1)  # Expand to time dimension
        x = torch.cat([code, s_emb], dim=-1)  # Concatenate in frequency dimension

        x, _ = self.lstm1(x)

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