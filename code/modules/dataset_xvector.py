import numpy as np
import torch
import pathlib

from modules import global_value as g


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_repeats, speaker_start=None, speaker_end=None, speech_start=None, speech_end=None):
        self.num_repeats = num_repeats

        speakers = sorted(list(pathlib.Path(g.mel_dir).glob('*')))
        speakers = speakers[speaker_start:speaker_end]
        self.speakers = speakers

        self.files = []
        for speaker in speakers:
            speeches = sorted(list(speaker.glob('*.pt')))
            speeches = speeches[speech_start:speech_end]

            self.files.append(speeches)

    def padding(self, data):
        if len(data) < g.seg_len:
            len_pad = g.seg_len - len(data)
            data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
        else:
            data = data[:g.seg_len]
        return data

    def __iter__(self):
        for i in range(self.num_repeats):
            np.random.seed(i)

            speaker_indices = np.random.choice(len(self.files), g.batch_size, replace=False)
            speech_indices = np.random.choice(len(self.files[0]), g.batch_size, replace=False)

            data = torch.stack([
                self.padding(torch.load(self.files[speaker_index][speech_index]))
                for speaker_index, speech_index in zip(speaker_indices, speech_indices)
            ], dim=0)

            speaker_indices = torch.from_numpy(speaker_indices).long()
            speech_indices = torch.from_numpy(speech_indices).long()
            yield data, (speaker_indices, speech_indices)
