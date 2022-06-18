import logging
import multiprocessing
import pathlib
import tempfile

import numpy as np
import torch

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

        self.set_seed(0)

    def __iter__(self):
        for _ in range(self.num_repeats):
            if self.use_same_speaker:
                speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=False)
                speech_indices  = self.rand_state.choice(len(self.files[0]), g.batch_size, replace=False)

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
                c_speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=False)
                s_speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=False)
                c_speech_indices  = self.rand_state.choice(len(self.files[0]), g.batch_size, replace=False)

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

    def set_seed(self, seed):
        self.rand_state = np.random.RandomState(seed)

    def padding(self, data, length):
        if len(data) < length:
            len_pad = length - len(data)
            data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
        else:
            data = data[:length]
        return data


class PnmDatasetStatic(torch.utils.data.Dataset):
    def __init__(self, num_repeats, speaker_start=None, speaker_end=None, phoneme_start=None, phoneme_end=None):
        self.num_repeats = num_repeats

        pnm_paths = sorted(pathlib.Path(g.pnm_dir).iterdir())
        pnm_paths = pnm_paths[speaker_start:speaker_end]

        self.data = []
        for pnm_path in pnm_paths:
            speaker_pnm = np.load(pnm_path, allow_pickle=True)
            speaker_pnm = speaker_pnm[phoneme_start:phoneme_end]
            self.data.append(speaker_pnm)

        self.set_seed(0)

    def __iter__(self):
        for _ in range(self.num_repeats):
            speaker_indices = self.rand_state.choice(len(self.data),    g.batch_size, replace=False)
            phoneme_indices = self.rand_state.choice(len(self.data[0]), g.batch_size, replace=False)

            data = torch.stack([
                self.padding(torch.from_numpy(self.data[speaker_index][phoneme_index]), g.pnm_len)
                for speaker_index, phoneme_index in zip(speaker_indices, phoneme_indices)
            ], dim=0)

            speaker_indices = torch.from_numpy(speaker_indices).long()
            phoneme_indices = torch.from_numpy(phoneme_indices).long()
            yield data, (speaker_indices, phoneme_indices)

    def set_seed(self, seed):
        self.rand_state = np.random.RandomState(seed)

    def padding(self, data, length):
        if len(data) < length:
            len_pad = length - len(data)
            data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
        else:
            data = data[:length]
        return data


class PnmDatasetDynamic(torch.utils.data.Dataset):
    def __init__(self, num_repeats, speaker_start=None, speaker_end=None, phoneme_start=None, phoneme_end=None):
        self.num_repeats = num_repeats
        self.phoneme_start = phoneme_start
        self.phoneme_end = phoneme_end

        org_pnm_paths = sorted(pathlib.Path(g.pnm_dir).iterdir())
        org_pnm_paths = org_pnm_paths[speaker_start:speaker_end]

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = pathlib.Path(self.tmp_dir.name)
        g.tmp_dirs.append(self.tmp_dir_path)
        logging.info(f'Created temporary directory {self.tmp_dir_path}')

        pool = multiprocessing.Pool(processes=g.processer_num)
        self.files = pool.map(self.load_pnm, org_pnm_paths)

        self.set_seed(0)

    def __iter__(self):
        for _ in range(self.num_repeats):
            speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=False)
            phoneme_indices = self.rand_state.choice(len(self.files[0]), g.batch_size, replace=False)

            data = torch.stack([
                self.padding(torch.load(self.files[speaker_index][phoneme_index]), g.pnm_len)
                for speaker_index, phoneme_index in zip(speaker_indices, phoneme_indices)
            ], dim=0)

            speaker_indices = torch.from_numpy(speaker_indices).long()
            phoneme_indices = torch.from_numpy(phoneme_indices).long()
            yield data, (speaker_indices, phoneme_indices)

    def __del__(self):
        self.tmp_dir.cleanup()

    def load_pnm(self, org_pnm_path):
        speaker_pnm = np.load(org_pnm_path, allow_pickle=True)
        speaker_pnm = speaker_pnm[self.phoneme_start:self.phoneme_end]

        pnm_dir = self.tmp_dir_path / org_pnm_path.stem
        pnm_dir.mkdir(exist_ok=True)

        for i, single_pnm in enumerate(speaker_pnm):
            torch.save(torch.from_numpy(single_pnm), pnm_dir / f'{i:06d}.pt')

        return sorted(pnm_dir.iterdir())

    def set_seed(self, seed):
        self.rand_state = np.random.RandomState(seed)

    def padding(self, data, length):
        if len(data) < length:
            len_pad = length - len(data)
            data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
        else:
            data = data[:length]
        return data
