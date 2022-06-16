import argparse
import logging
import pathlib
import traceback

import matplotlib; matplotlib.use('Agg')
import torch
import torch.utils.tensorboard

from modules import global_value as g
from modules import common
from modules import dataset
from modules import model
from modules import audio
from modules import vgg_perceptual_loss
from modules import ssim_loss


def main(config_path):
    common.custom_init(config_path, 'apple', '%Y%m/%d/%H%M%S')

    net = model.Net().to(g.device)
    logging.debug(f'MODEL: {net}')

    if g.model_load_path is not None:
        net.load_state_dict(torch.load(g.model_load_path, map_location=g.device))
        logging.debug(f'LOAD MODEL: {g.model_load_path}')

    if not g.no_train:
        train_dataset = dataset.Dataset(g.use_same_speaker, **g.train_dataset)
        valdt_dataset = dataset.Dataset(g.use_same_speaker, **g.valdt_dataset)
        tests_dataset = dataset.Dataset(g.use_same_speaker, **g.tests_dataset)

        vgg_criterion = vgg_perceptual_loss.VGGPerceptualLoss().to(g.device)
        ssim_criterion = ssim_loss.SSIMLoss(channel=1).to(g.device)

        def criterion(c, t, r, q, c_feat, q_feat):
            c = c.unsqueeze(1); t = t.unsqueeze(1); r = r.unsqueeze(1); q = q.unsqueeze(1)

            r_mse_loss = torch.nn.functional.mse_loss(r, t)
            r_vgg_loss = vgg_criterion(r, t)
            r_ssim_loss = ssim_criterion(r, t)
            r_loss = r_mse_loss + g.vgg_weight * r_vgg_loss + g.ssim_weight * r_ssim_loss

            q_mse_loss = torch.nn.functional.mse_loss(q, t)
            q_vgg_loss = vgg_criterion(q, t)
            q_ssim_loss = ssim_criterion(q, t)
            q_loss = q_mse_loss + g.vgg_weight * q_vgg_loss + g.ssim_weight * q_ssim_loss

            code_loss = torch.nn.functional.l1_loss(q_feat, c_feat)

            loss = r_loss + q_loss + code_loss

            losses = {
                'r_mse_loss': r_mse_loss, 'r_vgg_loss': r_vgg_loss, 'r_ssim_loss': r_ssim_loss, 'r_loss': r_loss,
                'q_mse_loss': q_mse_loss, 'q_vgg_loss': q_vgg_loss, 'q_ssim_loss': q_ssim_loss, 'q_loss': q_loss,
                'code_loss': code_loss, 'loss': loss,
            }

            return loss, losses

        (g.work_dir / 'cp').mkdir(parents=True)

        best_train_loss = best_valdt_loss = {'loss': float('inf')}

        with torch.utils.tensorboard.SummaryWriter(g.work_dir / 'tboard') as sw:
            total_epoch = 0

            for stage in g.stages:
                logging.info(f'STAGE: {stage}')

                if stage['optimizer'] == 'adam':
                    optimizer = torch.optim.Adam(net.parameters(), lr=stage['lr'])
                elif stage['optimizer'] == 'sgd':
                    optimizer = torch.optim.SGD(net.parameters(), lr=stage['lr'], momentum=stage['momentum'])
                logging.debug(f'SET OPTIMIZER: {optimizer}')

                try:
                    patience = 0

                    for epoch in range(stage['num_epochs']):
                        logging.info(f'EPOCH: {epoch + 1} (TOTAL: {total_epoch + 1})')

                        train_loss = model_train   (net, train_dataset, criterion, optimizer)
                        valdt_loss = model_validate(net, valdt_dataset, criterion)

                        logging.info(f'TRAIN LOSS: {train_loss["loss"]:.6f}, VALDT LOSS: {valdt_loss["loss"]:.6f}')

                        if train_loss['loss'] < best_train_loss['loss']:
                            best_train_loss = train_loss
                            torch.save(net.state_dict(), g.work_dir / 'cp' / 'best_train.pth')
                            logging.debug(f'SAVE BEST TRAIN MODEL: {g.work_dir / "cp" / "best_train.pth"}')

                        if valdt_loss['loss'] < best_valdt_loss['loss']:
                            best_valdt_loss = valdt_loss
                            torch.save(net.state_dict(), g.work_dir / 'cp' / 'best_valdt.pth')
                            logging.debug(f'SAVE BEST VALDT MODEL: {g.work_dir / "cp" / "best_valdt.pth"}')

                            patience = 0
                        else:
                            patience += 1

                        if patience >= stage['patience']:
                            logging.info(f'EARLY STOPPING: {patience}')
                            break

                        sw.add_scalars('train', train_loss, epoch)
                        sw.add_scalars('valdt', valdt_loss, epoch)
                        sw.flush()

                        total_epoch += 1
                except KeyboardInterrupt:
                    logging.info('SKIPPED BY USER')

                torch.save(net.state_dict(), g.work_dir / 'cp' / 'final.pth')
                torch.load(g.work_dir / 'cp' / 'best_valdt.pth', map_location=g.device)

                tests_loss = model_test(net, tests_dataset, criterion)

                logging.info(f'BEST TRAIN LOSS: {best_train_loss["loss"]:.6f}, BEST VALDT LOSS: {best_valdt_loss["loss"]:.6f}, TEST LOSS: {tests_loss["loss"]:.6f}')

    if g.need_predict:
        predict(net, **g.predict)


def model_train(net, dataset, criterion, optimizer):
    net.train()

    avg_losses = {}

    for i, (c, t, c_emb, s_emb, _) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)

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

        print(f'Training: {i + 1:03d}/{g.train_dataset["num_repeats"]:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')
    
    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.train_dataset['num_repeats'] * g.batch_size

    logging.debug(f'TRAIN LOSSES: {avg_losses}')

    return avg_losses


def model_validate(net, dataset, criterion):
    net.eval()

    avg_losses = {}

    for i, (c, t, c_emb, s_emb, _) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)

        with torch.no_grad():
            c_feat = net.content_enc(c, c_emb)
            r      = net.decoder(c_feat, s_emb)
            q      = r + net.postnet(r)
            q_feat = net.content_enc(q, c_emb)

        loss, losses = criterion(c, t, r, q, c_feat, q_feat)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Validate: {i + 1:03d}/{g.valdt_dataset["num_repeats"]:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.valdt_dataset['num_repeats'] * g.batch_size

    logging.debug(f'VALIDATE LOSSES: {avg_losses}')

    return avg_losses


def model_test(net, dataset, criterion):
    net.eval()

    avg_losses = {}

    for i, (c, t, c_emb, s_emb, _) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)

        with torch.no_grad():
            c_feat = net.content_enc(c, c_emb)
            r      = net.decoder(c_feat, s_emb)
            q      = r + net.postnet(r)
            q_feat = net.content_enc(q, c_emb)

        loss, losses = criterion(c, t, r, q, c_feat, q_feat)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Testing: {i + 1:03d}/{g.tests_dataset["num_repeats"]:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.tests_dataset['num_repeats'] * g.batch_size

    logging.debug(f'TEST LOSSES: {avg_losses}')

    return avg_losses


def predict(net, source_speaker, target_speaker, speech):
    net.eval()

    c = torch.load(f'{g.mel_dir}/{source_speaker}/{speech}.pt').unsqueeze(0).to(g.device)
    t = torch.load(f'{g.mel_dir}/{target_speaker}/{speech}.pt').unsqueeze(0).to(g.device)
    c_emb = torch.load(f'{g.emb_dir}/{source_speaker}.pt').unsqueeze(0).to(g.device)
    s_emb = torch.load(f'{g.emb_dir}/{target_speaker}.pt').unsqueeze(0).to(g.device)

    with torch.no_grad():
        c_feat = net.content_enc(c, c_emb)
        r      = net.decoder(c_feat, s_emb)
        q      = r + net.postnet(r)

    audio.save('source',     c.squeeze(0))
    audio.save('target',     t.squeeze(0))
    audio.save('rec_before', r.squeeze(0))
    audio.save('rec_after',  q.squeeze(0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=pathlib.Path, default='config.yml')

    try:
        main(**vars(parser.parse_args()))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()