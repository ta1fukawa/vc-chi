import torch

from modules import global_value as g

class DoNothing(torch.nn.Module):
    def __init__(self):
        super(DoNothing, self).__init__()

    def forward(self, x):
        return x

def get_activation(name, **kwargs):
    if name == 'relu':
        return torch.nn.ReLU(**kwargs)
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU(**kwargs)
    elif name == 'sigmoid':
        return torch.nn.Sigmoid(**kwargs)
    elif name == 'tanh':
        return torch.nn.Tanh(**kwargs)
    elif name == 'linear':
        return DoNothing()
    else:
        raise ValueError(f'Unknown activation: {name}')

class Layer(torch.nn.Module):
    def __init__(self, inp, oup, layer='linear', bn=False, activation=None, activation_param=None, **kwargs):
        super(Layer, self).__init__()

        if layer == 'linear':
            self.layer = torch.nn.Linear(inp, oup, **kwargs)
        elif layer == 'conv1d':
            self.layer = torch.nn.Conv1d(inp, oup, **kwargs)
        elif layer == 'conv2d':
            self.layer = torch.nn.Conv2d(inp, oup, **kwargs)

        if bn:
            if layer in ['linear', 'conv1d']:
                self.bn = torch.nn.BatchNorm1d(oup)
            elif layer in ['conv2d']:
                self.bn = torch.nn.BatchNorm2d(oup)
        else:
            self.bn = DoNothing()

        if activation is not None:
            self.activation = get_activation(activation)
            torch.nn.init.xavier_uniform_(self.layer.weight, gain=torch.nn.init.calculate_gain(activation, param=activation_param))
        else:
            self.activation = DoNothing()

    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
