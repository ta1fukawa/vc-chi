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
from modules import xvector
from modules import audio


def main(config_path):
    common.custom_init(config_path, '%Y%m%d/%H%M%S')

    net = xvector.Net().to(g.device)
    logging.debug(f'MODEL: {net}')

    if g.model_load_path is not None:
        net.load_state_dict(torch.load(g.model_load_path, map_location=g.device))
        logging.debug(f'LOAD MODEL: {g.model_load_path}')

    cross_entropy_loss = torch.nn.NLLLoss()
    def criterion(pred, indices):
        ce_loss = cross_entropy_loss(pred, indices)

        loss = ce_loss

        losses = {
            'loss': ce_loss,
        }

        return loss, losses

    (g.work_dir / 'cp').mkdir(parents=True)


    with torch.utils.tensorboard.SummaryWriter(g.work_dir / 'tboard') as sw:
        total_epoch = 0

        for stage_no, stage in enumerate(g.stages):
            logging.info(f'STAGE: {stage}')

            net.set_train_mode(stage['mode'])
            if stage['mode'] == 'small':
                ds = dataset.Dataset(g.use_same_speaker, **g.small_dataset)
                num_repeats = g.small_dataset['num_repeats']
            elif stage['mode'] == 'large':
                ds = dataset.Dataset(g.use_same_speaker, **g.large_dataset)
                num_repeats = g.large_dataset['num_repeats']

            if stage['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(net.parameters(), lr=stage['lr'])
            elif stage['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=stage['lr'], momentum=stage['momentum'])
            logging.debug(f'SET OPTIMIZER: {optimizer}')

            try:
                patience = 0

                best_train_loss = best_valdt_loss = {'loss': float('inf')}

                for epoch in range(stage['num_epochs']):
                    logging.info(f'EPOCH: {epoch + 1} (TOTAL: {total_epoch + 1})')

                    train_loss = model_train   (net, ds, criterion, optimizer, num_repeats)
                    valdt_loss = model_validate(net, ds, criterion, num_repeats)

                    logging.info(f'TRAIN LOSS: {train_loss["loss"]:.10f}, VALDT LOSS: {valdt_loss["loss"]:.10f}')

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

                    sw.add_scalars('train', train_loss, total_epoch)
                    sw.add_scalars('valdt', valdt_loss, total_epoch)
                    sw.flush()

                    total_epoch += 1
            except KeyboardInterrupt:
                logging.info('SKIPPED BY USER')

            torch.save(net.state_dict(), g.work_dir / 'cp' / f'{stage_no}_final.pth')
            torch.load(g.work_dir / 'cp' / f'{stage_no}_best_valdt.pth', map_location=g.device)

            tests_loss = model_test(net, ds, criterion, num_repeats)

            logging.info(f'BEST TRAIN LOSS: {best_train_loss["loss"]:.10f}, BEST VALDT LOSS: {best_valdt_loss["loss"]:.10f}, TEST LOSS: {tests_loss["loss"]:.10f}')


def model_train(net, dataset, criterion, optimizer, num_repeats):
    net.train()

    avg_losses = {}

    for i, (c, _, _, _, (speaker_indices, _, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        pred, _ = net(c)

        loss, losses = criterion(pred, speaker_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Training: {i + 1:03d}/{num_repeats:03d} (loss={loss.item() / g.batch_size:.10f})\033[K\033[G', end='')
    
    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= num_repeats * g.batch_size

    logging.debug(f'TRAIN LOSSES: {avg_losses}')

    return avg_losses


def model_validate(net, dataset, criterion, num_repeats):
    net.eval()

    avg_losses = {}

    for i, (c, _, _, _, (speaker_indices, _, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        with torch.no_grad():
            pred, _ = net(c)

        loss, losses = criterion(pred, speaker_indices)

        if loss.item() >= 100:
            net(c)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Validate: {i + 1:03d}/{num_repeats:03d} (loss={loss.item() / g.batch_size:.10f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= num_repeats * g.batch_size

    logging.debug(f'VALIDATE LOSSES: {avg_losses}')

    return avg_losses


def model_test(net, dataset, criterion, num_repeats):
    net.eval()

    avg_losses = {}

    for i, (c, _, _, _, (speaker_indices, _, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        with torch.no_grad():
            pred, _ = net(c)

        loss, losses = criterion(pred, speaker_indices)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Testing: {i + 1:03d}/{num_repeats:03d} (loss={loss.item() / g.batch_size:.10f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= num_repeats * g.batch_size

    logging.debug(f'TEST LOSSES: {avg_losses}')

    return avg_losses


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
