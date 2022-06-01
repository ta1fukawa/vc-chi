import numpy as np
import torch
import pathlib
from modules import global_value as g


class Dataset(torch.utils.data.Dataset):
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

        speakers = sorted(list(pathlib.Path('/home/g2181479/vc-beta/vc3/mel-jvs').glob('*')))
        speakers = speakers[:g.num_speakers] if test_mode else speakers[g.num_speakers:]

        self.files = []
        for speaker in speakers:
            speeches = sorted(list(speaker.glob('*.pt')))
            speeches = speeches[:g.num_speeches] if test_mode else speeches[g.num_speeches:]

            self.files.append(speeches)

    def padding(self, data):
        if len(data) < g.seg_len:
            len_pad = g.seg_len - len(data)
            data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
        else:
            data = data[:g.seg_len]
        return data

    def __iter__(self):
        for _ in range(g.num_repeats if not self.test_mode else g.num_test_repeats):
            c_speech_idxes  = np.random.choice(len(self.files[0]), g.batch_size, replace=False)
            s_speech_idxes  = np.random.choice(len(self.files[0]), g.batch_size, replace=False)
            c_speaker_idxes = np.random.choice(len(self.files), g.batch_size, replace=False)
            s_speaker_idxes = np.random.choice(len(self.files), g.batch_size, replace=False)
            c_data = torch.stack([
                self.padding(torch.load(self.files[speaker_idx][speech_idx]))
                for speaker_idx, speech_idx in zip(c_speaker_idxes, c_speech_idxes)
            ], dim=0)
            s_data = torch.stack([
                self.padding(torch.load(self.files[speaker_idx][speech_idx]))
                for speaker_idx, speech_idx in zip(s_speaker_idxes, s_speech_idxes)
            ], dim=0)
            t_data = torch.stack([
                self.padding(torch.load(self.files[speaker_idx][speech_idx]))
                for speaker_idx, speech_idx in zip(s_speaker_idxes, c_speech_idxes)
            ], dim=0)
            yield c_data, s_data, t_data