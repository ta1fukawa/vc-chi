import torch
import sys
import yaml
import pathlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import numpy as np

sys.path.append('code')

from modules import common
from modules import model
from modules import dataset
from modules import global_value as g


g.code_id = 'bear'
g.run_id = datetime.datetime.now().strftime('%Y%m/%d/%H%M%S')
g.device = torch.device('cuda:0')

work_dir = pathlib.Path('wd', g.code_id, g.run_id)
work_dir.mkdir(parents=True)

common.backup_codes(pathlib.Path(__file__).parent, work_dir / 'code')

config_path = pathlib.Path('config.yaml')
config = yaml.load(config_path.open(mode='r'), Loader=yaml.FullLoader)

for k, v in config.items():
    setattr(g, k, v)
g.batch_size = 1

net = model.Net().to(g.device)
net.load_state_dict(torch.load('model/model.pth', map_location=g.device))
net.eval()

ds = dataset.Dataset()
c, s, t = next(iter(ds))
c = c.to(g.device)
s = s.to(g.device)
t = t.to(g.device)

s=c
c_emb = net.style_enc(c)
s_emb = net.style_enc(s)
feat = net.content_enc(c, c_emb)
r = net.decoder(feat, s_emb)
q = r + net.postnet(r)

c = c.squeeze(0).detach().cpu().numpy()
s = s.squeeze(0).detach().cpu().numpy()
t = t.squeeze(0).detach().cpu().numpy()
r = r.squeeze(0).detach().cpu().numpy()
q = q.squeeze(0).detach().cpu().numpy()

a = [c, s, t, r, q]
c, s, t, r, q = ((a - np.min(a) + 1e-8) / np.max(a) * 255).astype(np.int32)

(work_dir / 'img').mkdir(parents=True)

plt.figure(figsize=(10, 6))
plt.imshow(c.T[::-1], cmap='hot')
plt.savefig(str(work_dir / 'img' / 'c.png'))

plt.figure(figsize=(10, 6))
plt.imshow(s.T[::-1], cmap='hot')
plt.savefig(str(work_dir / 'img' / 's.png'))

plt.figure(figsize=(10, 6))
plt.imshow(t.T[::-1], cmap='hot')
plt.savefig(str(work_dir / 'img' / 't.png'))

plt.figure(figsize=(10, 6))
plt.imshow(r.T[::-1], cmap='hot')
plt.savefig(str(work_dir / 'img' / 'r.png'))

plt.figure(figsize=(10, 6))
plt.imshow(q.T[::-1], cmap='hot')
plt.savefig(str(work_dir / 'img' / 'q.png'))
