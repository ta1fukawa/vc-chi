import torch
import torch.nn.functional


class DoNothing(torch.nn.Module):
    def __init__(self):
        super(DoNothing, self).__init__()

    def forward(self, x: torch.Tensor):
        return x

def get_activation(name, **kwargs):
    if name == 'relu':
        return torch.nn.ReLU(**kwargs)
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU(**kwargs)
    elif name == 'gelu':
        return torch.nn.GELU(**kwargs)
    elif name == 'sigmoid':
        return torch.nn.Sigmoid(**kwargs)
    elif name == 'tanh':
        return torch.nn.Tanh(**kwargs)
    elif name == 'linear':
        return DoNothing()
    else:
        raise ValueError(f'Unknown activation: {name}')

class Layer(torch.nn.Module):
    def __init__(self, inp, oup, layer='linear', bn=False, bn_first=False, activation='linear', weight_gain=None, activation_param=None, **kwargs):
        super(Layer, self).__init__()

        self.bn_first = bn_first

        if layer == 'linear':
            self.layer = torch.nn.Linear(inp, oup, **kwargs)
        elif layer == 'conv1d':
            self.layer = torch.nn.Conv1d(inp, oup, **kwargs)
        elif layer == 'conv2d':
            self.layer = torch.nn.Conv2d(inp, oup, **kwargs)

        if bn:
            size = inp if bn_first else oup
            if layer in ['linear', 'conv1d']:
                self.bn = torch.nn.BatchNorm1d(size)
            elif layer in ['conv2d']:
                self.bn = torch.nn.BatchNorm2d(size)
        else:
            self.bn = DoNothing()

        if weight_gain is None:
            if activation == 'gelu':
                weight_gain = 1
            else:
                weight_gain = torch.nn.init.calculate_gain(activation, param=activation_param)

        self.activation = get_activation(activation)
        torch.nn.init.xavier_uniform_(self.layer.weight, gain=weight_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bn_first:
            x = self.bn(x)
        x = self.layer(x)
        if not self.bn_first:
            x = self.bn(x)
        x = self.activation(x)
        return x

class TDNN(torch.nn.Module):
    def __init__(self, inp, oup, context_size=5, stride=1, dilation=1, dropout_p=0.2, batch_norm=False):
        super().__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = inp
        self.output_dim = oup
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = torch.nn.Linear(inp * context_size, oup)
        self.activation = torch.nn.ReLU()

        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(self.dropout_p)
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = torch.nn.functional.unfold(x, kernel_size=(self.context_size, self.input_dim), stride=(1, self.input_dim), dilation=(self.dilation, 1))
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.activation(x)

        if self.dropout_p > 0:
            x = self.dropout(x)
        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x