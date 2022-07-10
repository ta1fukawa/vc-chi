import argparse
import itertools
import logging
import math
import pathlib
import traceback
import csv

import numpy as np
import torch
import torch.utils.tensorboard

from modules import audio, common
from modules import dataset as ds
from modules import global_value as g
from modules import ssim_loss, vcmodel, vgg_perceptual_loss, xvector


def main(config_path, note):
    common.custom_init(config_path, '%Y%m/%d/%H%M%S', note)

    net = vcmodel.Net().to(g.device)
    logging.debug(f'MODEL: {net}')

    if g.model_load_path is not None:
        net.load_state_dict(torch.load(g.model_load_path, map_location=g.device))
        logging.debug(f'LOAD MODEL: {g.model_load_path}')

    train_dataset = ds.MelDataset(**g.train_dataset, speaker_size=g.speaker_size)
    valdt_dataset = ds.MelDataset(**g.valdt_dataset, speaker_size=g.speaker_size)
    tests_dataset = ds.MelDataset(**g.tests_dataset, speaker_size=g.speaker_size)

    vgg_criterion = vgg_perceptual_loss.VGGPerceptualLoss().to(g.device)
    sim_criterion = ssim_loss.SSIMLoss(channel=1).to(g.device)

    def criterion(c, t, r, q, c_feat, q_feat):
        c = c.unsqueeze(1); t = t.unsqueeze(1); r = r.unsqueeze(1); q = q.unsqueeze(1)

        r_mse_loss = torch.nn.functional.mse_loss(r, t)
        r_loss = r_mse_loss

        q_mse_loss = torch.nn.functional.mse_loss(q, t)
        q_vgg_loss = vgg_criterion(q, t)
        q_sim_loss = sim_criterion(q, t)
        q_loss = q_mse_loss + g.vgg_weight * q_vgg_loss + g.sim_weight * q_sim_loss

        code_loss = torch.nn.functional.l1_loss(q_feat, c_feat)

        loss = r_loss + q_loss + code_loss

        losses = {
            'r_mse_loss': r_mse_loss, 'r_loss': r_loss,
            'q_mse_loss': q_mse_loss, 'q_vgg_loss': q_vgg_loss, 'q_sim_loss': q_sim_loss, 'q_loss': q_loss,
            'code_loss': code_loss,
            'loss': loss,
        }

        return loss, losses

    (g.work_dir / 'cp').mkdir(parents=True)

    with torch.utils.tensorboard.SummaryWriter(g.work_dir / 'tboard') as sw:
        total_epoch = 0

        for stage_no, stage in enumerate(g.stages):
            logging.info(f'STAGE_{stage_no}: {stage}')
            common.update_note_status(f'stage_{stage_no}')

            if stage['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(net.parameters(), lr=stage['lr'])
            elif stage['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=stage['lr'], momentum=stage['momentum'])
            logging.debug(f'SET OPTIMIZER: {optimizer}')

            for dataset in [train_dataset, valdt_dataset, tests_dataset]:
                dataset.set_use_zero_emb(stage['use_zero_emb'])

            patience = 0

            best_train_loss = best_valdt_loss = {'loss': float('inf')}

            for epoch in range(stage['num_epochs']):
                logging.info(f'EPOCH: {epoch + 1} (TOTAL: {total_epoch + 1})')

                train_loss = model_train   (net, train_dataset, criterion, optimizer)
                valdt_loss = model_validate(net, valdt_dataset, criterion)

                logging.info(f'TRAIN LOSS: {train_loss["loss"]:.6f}, VALDT LOSS: {valdt_loss["loss"]:.6f}')

                if train_loss['loss'] < best_train_loss['loss']:
                    best_train_loss = train_loss
                    torch.save(net.state_dict(), g.work_dir / 'cp' / f'{stage_no}_best_train.pth')
                    logging.debug(f'SAVE BEST TRAIN MODEL: {g.work_dir / "cp" / "best_train.pth"}')

                if valdt_loss['loss'] < best_valdt_loss['loss']:
                    best_valdt_loss = valdt_loss
                    torch.save(net.state_dict(), g.work_dir / 'cp' / f'{stage_no}_best_valdt.pth')
                    logging.debug(f'SAVE BEST VALDT MODEL: {g.work_dir / "cp" / "best_valdt.pth"}')

                    patience = 0
                else:
                    patience += 1

                if patience >= stage['patience']:
                    logging.info(f'EARLY STOPPING: {patience}')
                    break


                for key in train_loss.keys():
                    sw.add_scalars(key, {'train': train_loss[key], 'valdt': valdt_loss[key]}, total_epoch)
                sw.flush()

                total_epoch += 1

            torch.save(net.state_dict(), g.work_dir / 'cp' / f'{stage_no}_final.pth')

            if (g.work_dir / 'cp' / f'{stage_no}_best_valdt.pth').exists():
                torch.load(g.work_dir / 'cp' / f'{stage_no}_best_valdt.pth', map_location=g.device)
                logging.debug(f'LOAD BEST VALDT MODEL: {g.work_dir / "cp" / "best_valdt.pth"}')

            tests_loss_same = model_test(net, tests_dataset, criterion, True)
            tests_loss_diff = model_test(net, tests_dataset, criterion, False)

            logging.info(f'BEST TRAIN LOSS: {best_train_loss["loss"]:.6f}, BEST VALDT LOSS: {best_valdt_loss["loss"]:.6f}, SAME TEST LOSS: {tests_loss_same["loss"]:.6f}, DIFF TEST LOSS: {tests_loss_diff["loss"]:.6f}')

    predict(net)

    logging.info(f'DONE: {g.code_id}/{g.run_id} {note}')


def model_train(net, dataset, criterion, optimizer):
    net.train()

    avg_losses = {}

    for i, (c, t, c_emb, s_emb, _) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)
        c_emb = c_emb.to(g.device); s_emb = s_emb.to(g.device)

        c_feat = net.content_enc(c, c_emb)
        r      = net.decoder(c_feat, s_emb)
        q      = r + net.postnet(r)
        q_feat = net.content_enc(q, c_emb)

        loss, losses = criterion(c, t, r, q, c_feat, q_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Training: {i + 1:03d}/{g.train_dataset["num_repeats"]:03d} (loss={loss.item():.6f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.train_dataset['num_repeats']

    logging.debug(f'TRAIN LOSSES: {avg_losses}')

    return avg_losses


def model_validate(net, dataset, criterion):
    net.eval()

    avg_losses = {}

    for i, (c, t, c_emb, s_emb, _) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)
        c_emb = c_emb.to(g.device); s_emb = s_emb.to(g.device)

        with torch.no_grad():
            c_feat = net.content_enc(c, c_emb)
            r      = net.decoder(c_feat, s_emb)
            q      = r + net.postnet(r)
            q_feat = net.content_enc(q, c_emb)

        loss, losses = criterion(c, t, r, q, c_feat, q_feat)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Validate: {i + 1:03d}/{g.valdt_dataset["num_repeats"]:03d} (loss={loss.item():.6f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.valdt_dataset['num_repeats']

    logging.debug(f'VALIDATE LOSSES: {avg_losses}')

    return avg_losses


def model_test(net, dataset, criterion, use_same_speaker):
    net.eval()
    dataset.set_seed(0)
    dataset.use_same_speaker = use_same_speaker

    avg_losses = {}

    for i, (c, t, c_emb, s_emb, _) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)
        c_emb = c_emb.to(g.device); s_emb = s_emb.to(g.device)

        with torch.no_grad():
            c_feat = net.content_enc(c, c_emb)
            r      = net.decoder(c_feat, s_emb)
            q      = r + net.postnet(r)
            q_feat = net.content_enc(q, c_emb)

        loss, losses = criterion(c, t, r, q, c_feat, q_feat)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Testing: {i + 1:03d}/{g.tests_dataset["num_repeats"]:03d} (loss={loss.item():.6f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.tests_dataset['num_repeats']

    logging.debug(f'TEST LOSSES: {avg_losses}')

    return avg_losses


def predict(net):
    net.eval()

    dataset = ds.MelDataset(0, speaker_size=None, speech_start=-g.batch_size)

    speaker_indices = np.arange(g.pred_num_speakers)
    speech_indices  = np.zeros(g.pred_num_speakers, dtype=np.int)
    data = dataset.load_data(speaker_indices, speech_indices)

    for i in range(g.pred_num_speakers):
        audio.save(f'org/{i + 1:03d}', data[i].squeeze(1))

    c_speaker_indices = np.arange(g.batch_size) % g.speaker_size
    s_speaker_indices = np.arange(g.batch_size) % g.speaker_size
    speech_indices = np.arange(g.batch_size)

    mse_mat = np.zeros((g.pred_num_speakers, g.pred_num_speakers))

    target_speakers = np.concatenate((np.arange(g.pred_num_speakers // 2), (len(dataset.files) - g.pred_num_speakers // 2 + np.arange(g.pred_num_speakers // 2))))

    for i in range(g.pred_num_speakers):
        for j in range(g.pred_num_speakers):
            c_speaker_indices[0] = target_speakers[i]
            s_speaker_indices[0] = target_speakers[j]

            c = dataset.load_data(c_speaker_indices, speech_indices)
            t = dataset.load_data(s_speaker_indices, speech_indices)
            c_emb = dataset.load_emb(c_speaker_indices)
            s_emb = dataset.load_emb(s_speaker_indices)

            c = c.to(g.device); t = t.to(g.device)
            c_emb = c_emb.to(g.device); s_emb = s_emb.to(g.device)

            with torch.no_grad():
                c_feat = net.content_enc(c, c_emb)
                r = net.decoder(c_feat, s_emb)
                q = r + net.postnet(r)

            mse_loss = torch.nn.functional.mse_loss(q[:1], t[:1])
            mse_mat[i, j] = mse_loss.item()

            audio.save(f'{i + 1:03d}to{j + 1:03d}', q.squeeze(1)[0])

    with open(g.work_dir / f'mse_mat.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(mse_mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=pathlib.Path, default='./vc_config.yml')
    parser.add_argument('-n', '--note', type=str, default=None)

    try:
        main(**vars(parser.parse_args()))
        common.update_note_status('done')
    except Exception as e:
        logging.error(traceback.format_exc())
        common.update_note_status('error')
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()
