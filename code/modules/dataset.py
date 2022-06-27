import csv
import multiprocessing
import pathlib

import librosa
import numpy as np
import torch

from modules import audio
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
                speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=True)
                speech_indices  = self.rand_state.choice(len(self.files[0]), g.batch_size, replace=True)

                data = torch.stack([
                    padding(torch.load(self.files[speaker_index][speech_index]), g.seg_len)
                    for speaker_index, speech_index in zip(speaker_indices, speech_indices)
                ], dim=0)

                emb = torch.stack([
                    torch.load(pathlib.Path(g.emb_dir) / f'{self.speakers[speaker_index].name}.pt')
                    for speaker_index in speaker_indices
                ], dim=0)
                # emb = torch.zeros_like(emb)

                speaker_indices = torch.from_numpy(speaker_indices).long()
                speech_indices  = torch.from_numpy(speech_indices).long()
                yield data, data, emb, emb, (speaker_indices, speech_indices, speaker_indices)
            else:
                c_speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=True)
                s_speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=True)
                c_speech_indices  = self.rand_state.choice(len(self.files[0]), g.batch_size, replace=True)

                c_data = torch.stack([
                    padding(torch.load(self.files[speaker_index][speech_index]), g.seg_len)
                    for speaker_index, speech_index in zip(c_speaker_indices, c_speech_indices)
                ], dim=0)
                t_data = torch.stack([
                    padding(torch.load(self.files[speaker_index][speech_index]), g.seg_len)
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
                # c_emb = torch.zeros_like(c_emb)
                # s_emb = torch.zeros_like(s_emb)

                c_speaker_indices = torch.from_numpy(c_speaker_indices).long()
                c_speech_indices  = torch.from_numpy(c_speech_indices).long()
                s_speaker_indices = torch.from_numpy(s_speaker_indices).long()
                yield c_data, t_data, c_emb, s_emb, (c_speaker_indices, c_speech_indices, s_speaker_indices)

    def set_seed(self, seed):
        self.rand_state = np.random.RandomState(seed)


class PnmDataset(torch.utils.data.Dataset):
    def __init__(self, speaker_size, num_repeats, phoneme_start=None, phoneme_end=None):
        self.num_repeats = num_repeats

        speakers = sorted(pathlib.Path(g.wav_dir).iterdir())
        speakers = speakers[:speaker_size]

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.waves = pool.map(load_speaker_waves, zip(speakers, [phoneme_start, phoneme_end] * len(speakers)))
        pool.close(); pool.join()

        self.set_seed(0)

    def __iter__(self):
        for _ in range(self.num_repeats):
            speaker_indices = self.rand_state.choice(len(self.waves),    g.batch_size, replace=True)
            phoneme_indices = self.rand_state.choice(len(self.waves[0]), g.batch_size, replace=True)

            data = torch.stack([
                padding(torch.from_numpy(audio.fast_stft(self.waves[speaker_index][phoneme_index]).T[1:]), g.pad_pnm_len)
                for speaker_index, phoneme_index in zip(speaker_indices, phoneme_indices)
            ], dim=0)

            speaker_indices = torch.from_numpy(speaker_indices).long()
            phoneme_indices = torch.from_numpy(phoneme_indices).long()
            yield data, (speaker_indices, phoneme_indices)

    def set_seed(self, seed):
        self.rand_state = np.random.RandomState(seed)


def load_speaker_waves(speaker, phoneme_start=None, phoneme_end=None):
    speaker_waves = []

    flat_labs = []
    for lab_path in sorted((pathlib.Path(g.lab_dir) / speaker.name).iterdir()):
        with lab_path.open('r') as f:
            reader = csv.reader(f, delimiter='\t')
            for start_sec, end_sec, phoneme in reader:
                if phoneme in ['silB', 'silE', 'sp']:
                    continue

                start_sample = int(float(start_sec) * g.sample_rate)
                end_sample   = int(float(end_sec)   * g.sample_rate)

                if (end_sample - start_sample - g.fft_size) / g.hop_size + 1 < g.min_pnm_len:
                    continue

                flat_labs.append((lab_path.stem, start_sample, end_sample))

    flat_labs = flat_labs[phoneme_start:phoneme_end]

    tree_labs = []
    for lab_path in sorted((pathlib.Path(g.lab_dir) / speaker.name).iterdir()):
        labs = [(start_sample, end_sample) for lab_name, start_sample, end_sample in flat_labs if lab_name == lab_path.stem]
        if len(labs) > 0:
            tree_labs.append((lab_path.stem, labs))

    speaker_waves = []
    for lab_name, labs in tree_labs:
        wave, sr = librosa.load(speaker / f'{lab_name}.wav', sr=g.sample_rate)

        for start_sample, end_sample in labs:
            speaker_waves.append(wave[start_sample:end_sample])

    return speaker_waves


def padding(data, length):
    if len(data) < length:
        len_pad = length - len(data)
        data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
    else:
        data = data[:length]
    return data
