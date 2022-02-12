import torch
import sys

sys.path.append('code')

from modules import global_value as g
from modules import model

g.nmels = 80
g.style_dim = 256
g.dim_neck = 16
g.lstm_stride = 16

BS = 2
SEQ_LEN = 128

net = model.Net().to('cuda')
print(net)

c = torch.randn(BS, SEQ_LEN, g.nmels).to('cuda')
print(c.shape)  # shape = (BS, SEQ_LEN, nmels)
s = torch.randn(BS, SEQ_LEN, g.nmels).to('cuda')
print(s.shape)  # shape = (BS, SEQ_LEN, nmels)

c_emb = net.style_enc(c)
print(c_emb.shape)  # shape = (BS, dim)
s_emb = net.style_enc(s)
print(s_emb.shape)  # shape = (BS, dim)

code = net.content_enc(c, c_emb)
print(code.shape)  # shape = (BS, SEQ_LEN, dim_neck * 2)

r = net.decoder(code, s_emb)
print(r.shape)  # shape = (BS, SEQ_LEN, nmels)
