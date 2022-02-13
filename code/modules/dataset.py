import numpy as np
import torch
import pathlib
from modules import global_value as g

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.files = sorted(list(pathlib.Path('/home/g2181479/vc-beta/vc3/mel-jvs').glob('*/*.pt')))

    def padding(self, data):
        if len(data) < g.seg_len:
            len_pad = g.seg_len - len(data)
            data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
        else:
            data = data[:g.seg_len]
        return data

    def __iter__(self):
        for _ in range(g.num_repeats):
            c_files = np.random.choice(self.files, size=g.batch_size, replace=False)
            c_data = [self.padding(torch.load(f)) for f in c_files]
            c_data = torch.stack(c_data, dim=0)

            s_files = np.random.choice(self.files, size=g.batch_size, replace=False)
            s_data = [self.padding(torch.load(f)) for f in s_files]
            s_data = torch.stack(s_data, dim=0)
            yield c_data, s_data
