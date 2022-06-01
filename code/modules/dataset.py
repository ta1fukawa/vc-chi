import numpy as np
import torch
import pathlib
from modules import global_value as g


class Dataset(torch.utils.data.Dataset):
    def __init__(self, use_same_speaker, num_repeats, speaker_start=None, speaker_end=None, speech_start=None, speech_end=None):
        self.use_same_speaker = use_same_speaker
        self.num_repeats = num_repeats

        speakers = sorted(list(pathlib.Path(g.mel_path).glob('*')))
        speakers = speakers[speaker_start:speaker_end]

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

            if self.use_same_speaker:
                speech_idxes = np.random.choice(len(self.files[0]), g.batch_size, replace=False)
                speaker_idxes = np.random.choice(len(self.files), g.batch_size, replace=False)

                data = torch.stack([
                    self.padding(torch.load(self.files[speaker_idx][speech_idx]))
                    for speaker_idx, speech_idx in zip(speaker_idxes, speech_idxes)
                ], dim=0)

                yield data, data, data
            else:
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
