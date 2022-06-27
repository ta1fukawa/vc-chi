import argparse
from cgi import print_arguments
import csv
import logging
import pathlib
import traceback

import librosa
import torch

from modules import audio, common
from modules import global_value as g
from modules import xvector


def main(config_path):
    common.custom_init(config_path, '%Y%m%d/%H%M%S')

    net = xvector.Net().to(g.device)
    logging.debug(f'MODEL: {net}')

    if g.model_load_path is not None:
        net.load_state_dict(torch.load(g.model_load_path, map_location=g.device))
        logging.info(f'LOAD MODEL: {g.model_load_path}')
    else:
        raise Exception('model_load_path is None')

    predict(net)


def predict(net):
    net.eval()

    wav_dir = pathlib.Path(g.bak_dir)
    emb_dir = pathlib.Path(g.emb_dir)

    emb_dir.mkdir(parents=True, exist_ok=True)

    for speaker in sorted(wav_dir.iterdir()):
        if not speaker.is_dir():
            continue

        try:
            speaker_waves = load_wav(speaker)
        except:
            print(f'{speaker.name} is not found')
            continue
        speaker_mels = [
            padding(torch.from_numpy(audio.fast_stft(wave).T[1:]), g.pad_pnm_len)
            for wave in speaker_waves
        ]

        if len(speaker_mels) > 0:
            speaker_mels = torch.stack(speaker_mels, dim=0).to(g.device)
            embs = []
            for i in range((speaker_mels.shape[0] - 1) // g.batch_size + 1):
                with torch.no_grad():
                    _, emb = net(speaker_mels[i * g.batch_size: (i + 1) * g.batch_size])
                    embs.append(emb)
            embs = torch.cat(embs, dim=0)
            emb = torch.mean(embs, dim=0).cpu()
            torch.save(emb, str(emb_dir / f'{speaker.name}.pt'))

    print('\033[K\033[G', end='')


def load_wav(speaker):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=pathlib.Path, default='xvector_config.yml')

    try:
        main(**vars(parser.parse_args()))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()
