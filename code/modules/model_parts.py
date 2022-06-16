import torch
import torch.nn.functional

from modules import global_value as g


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
    def __init__(self, inp, oup, layer='linear', bn=False, activation='linear', activation_param=None, **kwargs):
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

        self.activation = get_activation(activation)
        torch.nn.init.xavier_uniform_(self.layer.weight, gain=torch.nn.init.calculate_gain(activation, param=activation_param))

    def forward(self, x: torch.Tensor):
        x = self.layer(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class TDNN(torch.nn.Module):
    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1, batch_norm=False, dropout_p=0.2):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = torch.nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = torch.nn.ReLU()
        if self.batch_norm:
            self.bn = torch.nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = torch.nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        x = torch.nn.functional.unfold(x, (self.context_size, self.input_dim), stride=(1,self.input_dim), dilation=(self.dilation,1))

        x = x.transpose(1,2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x
