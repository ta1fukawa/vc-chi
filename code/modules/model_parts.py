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

    def forward(self, x: torch.Tensor):
        if self.bn_first:
            x = self.bn(x)
        x = self.layer(x)
        if not self.bn_first:
            x = self.bn(x)
        x = self.activation(x)
        return x
