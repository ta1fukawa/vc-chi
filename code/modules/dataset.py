import csv
import multiprocessing
import pathlib

import librosa
import numpy as np
import torch

from modules import audio
from modules import global_value as g


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, num_repeats, speaker_start=None, speaker_end=None, speech_start=None, speech_end=None):
        self.use_same_speaker = g.use_same_speaker
        self.num_repeats = num_repeats

        speaker_start = g.speaker_start if speaker_start is None else speaker_start
        speaker_end   = g.speaker_end   if speaker_end   is None else speaker_end

        speakers = sorted(pathlib.Path(g.mel_dir).iterdir())
        speakers = speakers[speaker_start:speaker_end]
        self.speakers = speakers

        self.files = []
        for speaker in speakers:
            speeches = sorted(speaker.iterdir())
            speeches = speeches[speech_start:speech_end]

            self.files.append(speeches)

        self.set_seed(0)
        self.set_use_zero_emb(False)

    def __iter__(self):
        for _ in range(self.num_repeats):
            if self.use_same_speaker:
                speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=True)
                speech_indices  = self.rand_state.choice(len(self.files[0]), g.batch_size, replace=True)

                data = self.load_data(speaker_indices, speech_indices)
                emb  = self.load_emb(speaker_indices)

                speaker_indices = torch.from_numpy(speaker_indices).long()
                speech_indices  = torch.from_numpy(speech_indices) .long()

                yield data, data, emb, emb, (speaker_indices, speech_indices, speaker_indices)
            else:
                c_speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=True)
                s_speaker_indices = self.rand_state.choice(len(self.files),    g.batch_size, replace=True)
                c_speech_indices  = self.rand_state.choice(len(self.files[0]), g.batch_size, replace=True)

                c_data = self.load_data(c_speaker_indices, c_speech_indices)
                t_data = self.load_data(s_speaker_indices, c_speech_indices)
                c_emb  = self.load_emb(c_speaker_indices)
                s_emb  = self.load_emb(s_speaker_indices)

                c_speaker_indices = torch.from_numpy(c_speaker_indices).long()
                c_speech_indices  = torch.from_numpy(c_speech_indices) .long()
                s_speaker_indices = torch.from_numpy(s_speaker_indices).long()

                yield c_data, t_data, c_emb, s_emb, (c_speaker_indices, c_speech_indices, s_speaker_indices)

    def set_seed(self, seed=0):
        self.rand_state = np.random.RandomState(seed)

    def set_use_zero_emb(self, use_zero_emb):
        self.use_zero_emb = use_zero_emb

    def load_data(self, speaker_indices, speech_indices):
        data = torch.stack([
            padding(torch.load(self.files[speaker_index][speech_index]), g.seg_len)
            for speaker_index, speech_index in zip(speaker_indices, speech_indices)
        ], dim=0)

        return data

    def load_emb(self, speaker_indices):
        emb = torch.stack([
            torch.load(pathlib.Path(g.emb_dir) / f'{self.speakers[speaker_index].name}.pt')
            for speaker_index in speaker_indices
        ], dim=0)

        if self.use_zero_emb:
            emb = torch.zeros_like(emb)
        else:
            if g.embed_type == 'emb':
                pass
            elif g.embed_type == 'label':
                emb = torch.from_numpy(speaker_indices).unsqueeze(1).expand(-1, emb.shape[1])

        return emb


class PnmDataset(torch.utils.data.Dataset):
    def __init__(self, speaker_size, num_repeats, phoneme_start=None, phoneme_end=None):
        self.num_repeats = num_repeats

        speakers = sorted(pathlib.Path(g.wav_dir).iterdir())
        speakers = speakers[:speaker_size]

        self.waves = []
        for speaker in speakers:
            speaker_waves = load_speaker_waves(speaker, phoneme_start, phoneme_end)
            if speaker_waves is not None:
                self.waves.append(speaker_waves)
                if len(self.waves) >= speaker_size:
                    break
        else:
            raise ValueError('Speaker size is too large.')

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
    for wav_path in sorted(speaker.iterdir()):
        with pathlib.Path(g.lab_dir, speaker.name, f'{wav_path.stem}.lab').open('r') as f:
            reader = csv.reader(f, delimiter='\t')
            for start_sec, end_sec, phoneme in reader:
                if phoneme in ['silB', 'silE', 'sp']:
                    continue

                start_sample = int(float(start_sec) * g.sample_rate)
                end_sample   = int(float(end_sec)   * g.sample_rate)

                if (end_sample - start_sample - g.fft_size) / g.hop_size + 1 < g.min_pnm_len:
                    continue

                flat_labs.append((wav_path.name, start_sample, end_sample))

    if phoneme_start is not None and phoneme_end is not None:
        if len(flat_labs) < phoneme_end - phoneme_start:
            return None

        flat_labs = flat_labs[phoneme_start:phoneme_end]

    tree_labs = []
    for wav_path in sorted(speaker.iterdir()):
        labs = [(start_sample, end_sample) for wav_path_name, start_sample, end_sample in flat_labs if wav_path_name == wav_path.name]
        if len(labs) > 0:
            tree_labs.append((wav_path, labs))

    speaker_waves = []
    for wav_path, labs in tree_labs:
        wave, sr = librosa.load(wav_path, sr=g.sample_rate)

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
