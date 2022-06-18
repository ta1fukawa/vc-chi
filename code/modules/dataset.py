import numpy as np
import torch
import pathlib

from modules import global_value as g


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, use_same_speaker, num_repeats, speaker_start=None, speaker_end=None, speech_start=None, speech_end=None):
        self.use_same_speaker = use_same_speaker
        self.num_repeats = num_repeats

        speakers = sorted(pathlib.Path(g.mel_dir).iterdir())
        speakers = speakers[speaker_start:speaker_end]
        self.speakers = speakers

        self.files = []
        for speaker in speakers:
            speeches = sorted(speaker.iterdir())
            speeches = speeches[speech_start:speech_end]

            self.files.append(speeches)

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

            if self.use_same_speaker:
                speaker_indices = np.random.choice(len(self.files),    g.batch_size, replace=False)
                speech_indices  = np.random.choice(len(self.files[0]), g.batch_size, replace=False)

                data = torch.stack([
                    self.padding(torch.load(self.files[speaker_index][speech_index]), g.seg_len)
                    for speaker_index, speech_index in zip(speaker_indices, speech_indices)
                ], dim=0)

                emb = torch.stack([
                    torch.load(pathlib.Path(g.emb_dir) / f'{self.speakers[speaker_index].name}.pt')
                    for speaker_index in speaker_indices
                ], dim=0)

                speaker_indices = torch.from_numpy(speaker_indices).long()
                speech_indices  = torch.from_numpy(speech_indices).long()
                yield data, data, emb, emb, (speaker_indices, speech_indices, speaker_indices)
            else:
                c_speaker_indices = np.random.choice(len(self.files),    g.batch_size, replace=False)
                s_speaker_indices = np.random.choice(len(self.files),    g.batch_size, replace=False)
                c_speech_indices  = np.random.choice(len(self.files[0]), g.batch_size, replace=False)

                c_data = torch.stack([
                    self.padding(torch.load(self.files[speaker_index][speech_index]), g.seg_len)
                    for speaker_index, speech_index in zip(c_speaker_indices, c_speech_indices)
                ], dim=0)
                t_data = torch.stack([
                    self.padding(torch.load(self.files[speaker_index][speech_index]), g.seg_len)
                    for speaker_index, speech_index in zip(s_speaker_indices, c_speech_indices)
                ], dim=0)

                c_emb = torch.stack([
                    torch.load(pathlib.Path(g.emb_dir) / f'{self.speakers[speaker_index].name}.pt')
                    for speaker_index in c_speaker_indices
                ], dim=0)
                s_emb = torch.stack([
                    torch.load(pathlib.Path(g.emb_dir) / f'{self.speakers[speaker_index].name}.pt')
                    for speaker_index in s_speaker_indices
                ], dim=0)

                c_speaker_indices = torch.from_numpy(c_speaker_indices).long()
                c_speech_indices  = torch.from_numpy(c_speech_indices).long()
                s_speaker_indices = torch.from_numpy(s_speaker_indices).long()
                yield c_data, t_data, c_emb, s_emb, (c_speaker_indices, c_speech_indices, s_speaker_indices)


class PnmDataset(torch.utils.data.Dataset):
    def __init__(self, num_repeats, speaker_start=None, speaker_end=None, phoneme_start=None, phoneme_end=None):
        self.num_repeats = num_repeats

        pnm_paths = sorted(pathlib.Path(g.pnm_dir).iterdir())
        pnm_paths = pnm_paths[speaker_start:speaker_end]

        self.data = []
        for pnm_path in pnm_paths:
            speaker_pnm = np.load(pnm_path, allow_pickle=True)
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
        for _ in range(self.num_repeats):
            speaker_indices = np.random.choice(len(self.data),    g.batch_size, replace=False)
            phoneme_indices = np.random.choice(len(self.data[0]), g.batch_size, replace=False)

            data = torch.stack([
                self.padding(torch.from_numpy(self.data[speaker_index][phoneme_index]), g.pnm_len)
                for speaker_index, phoneme_index in zip(speaker_indices, phoneme_indices)
            ], dim=0)

            speaker_indices = torch.from_numpy(speaker_indices).long()
            phoneme_indices = torch.from_numpy(phoneme_indices).long()
            yield data, (speaker_indices, phoneme_indices)

    def sed_seed(self, seed):
        np.random.seed(seed)
