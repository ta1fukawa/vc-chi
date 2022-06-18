import numpy as np
import torch
import pathlib

from modules import global_value as g


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_repeats, speaker_start=None, speaker_end=None, phoneme_start=None, phoneme_end=None):
        self.num_repeats = num_repeats

        pnm_paths = sorted(pathlib.Path(g.pnm_dir).iterdir())
        pnm_paths = pnm_paths[speaker_start:speaker_end]

        self.data = []
        for pnm_path in pnm_paths:
            speaker_pnm = np.load(pnm_path)
            speaker_pnm = speaker_pnm[phoneme_start:phoneme_end]
            self.data.append(speaker_pnm)

    def padding(self, data, length):
        if len(data) < length:
            len_pad = length - len(data)
            data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
        else:
            data = data[:length]
        return data

    def __iter__(self):
        for i in range(self.num_repeats):
            np.random.seed(i)

            speaker_indices = np.random.choice(len(self.data),    g.batch_size, replace=False)
            phoneme_indices = np.random.choice(len(self.data[0]), g.batch_size, replace=False)

            data = torch.stack([
                self.padding(torch.from_numpy(self.data[speaker_index][phoneme_index]), g.pnm_len)
                for speaker_index, phoneme_index in zip(speaker_indices, phoneme_indices)
            ], dim=0)

            speaker_indices = torch.from_numpy(speaker_indices).long()
            phoneme_indices = torch.from_numpy(phoneme_indices).long()
            yield data, (speaker_indices, phoneme_indices)
