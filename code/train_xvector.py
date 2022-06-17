import argparse
import logging
import pathlib
import traceback

import matplotlib; matplotlib.use('Agg')
import torch
import torch.utils.tensorboard

from modules import global_value as g
from modules import common
from modules import dataset_xvector
from modules import xvector
from modules import audio


def main(config_path):
    common.custom_init(config_path, '%Y%m%d/%H%M%S')

    net = xvector.Net2().to(g.device)
    logging.debug(f'MODEL: {net}')

    if g.model_load_path is not None:
        net.load_state_dict(torch.load(g.model_load_path, map_location=g.device))
        logging.debug(f'LOAD MODEL: {g.model_load_path}')

    nll_criterion = torch.nn.NLLLoss()
    def criterion(pred, indices):
        nll_loss = nll_criterion(pred, indices)

        loss = nll_loss

        losses = {
            'nll_loss': nll_loss,
            'loss': loss,
        }

        return loss, losses

    def accuracy(pred, indices):
        acc = (pred.argmax(dim=1) == indices).float().mean()

        return acc

    (g.work_dir / 'cp').mkdir(parents=True)

    with torch.utils.tensorboard.SummaryWriter(g.work_dir / 'tboard') as sw:
        total_epoch = 0

        for stage_no, stage in enumerate(g.stages):
            logging.info(f'STAGE: {stage}')

            net.set_train_mode(stage['mode'])
            if stage['mode'] == 'small':
                ds = dataset_xvector.Dataset(**g.small_dataset)
                num_repeats = g.small_dataset['num_repeats']
            elif stage['mode'] == 'large':
                ds = dataset_xvector.Dataset(**g.large_dataset)
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

                    train_loss, train_accuracy = model_train   (net, ds, criterion, accuracy, optimizer, num_repeats)
                    valdt_loss, valdt_accuracy = model_validate(net, ds, criterion, accuracy, num_repeats)

                    logging.info(f'TRAIN LOSS: {train_loss["loss"]:.10f}, VALDT LOSS: {valdt_loss["loss"]:.10f}')
                    logging.info(f'TRAIN ACC: {train_accuracy * 100:.4f}, VALDT ACC: {valdt_accuracy * 100:.4f}')

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

            tests_loss, tests_accuracy = model_test(net, ds, criterion, accuracy, num_repeats)

            logging.info(f'BEST TRAIN LOSS: {best_train_loss["loss"]:.10f}, BEST VALDT LOSS: {best_valdt_loss["loss"]:.10f}, TEST LOSS: {tests_loss["loss"]:.10f}')


def model_train(net, dataset, criterion, accuracy, optimizer, num_repeats):
    net.train()

    avg_losses = {}
    avg_acc = 0.0

    for i, (c, (speaker_indices, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        pred, _ = net(c)

        loss, losses = criterion(pred, speaker_indices)
        acc = accuracy(pred, speaker_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        avg_acc += acc.item()

        print(f'Training: {i + 1:03d}/{num_repeats:03d} (loss={loss.item() / g.batch_size:.10f}, acc={acc.item() * 100:.4f})\033[K\033[G', end='')
    
    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= num_repeats * g.batch_size
    
    avg_acc /= num_repeats * g.batch_size

    logging.debug(f'TRAIN LOSSES: {avg_losses}')
    logging.debug(f'TRAIN ACC: {avg_acc * 100}')

    return avg_losses, avg_acc


def model_validate(net, dataset, criterion, accuracy, num_repeats):
    net.eval()

    avg_losses = {}
    avg_acc = 0.0

    for i, (c, (speaker_indices, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        with torch.no_grad():
            pred, _ = net(c)

        loss, losses = criterion(pred, speaker_indices)
        acc = accuracy(pred, speaker_indices)

        if loss.item() >= 100:
            net(c)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()
        
        avg_acc += acc.item()

        print(f'Validate: {i + 1:03d}/{num_repeats:03d} (loss={loss.item() / g.batch_size:.10f}, acc={acc.item() * 100:.4f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= num_repeats * g.batch_size

    avg_acc /= num_repeats * g.batch_size

    logging.debug(f'VALIDATE LOSSES: {avg_losses}')
    logging.debug(f'VALIDATE ACC: {avg_acc * 100}')

    return avg_losses, avg_acc


def model_test(net, dataset, criterion, accuracy, num_repeats):
    net.eval()

    avg_losses = {}
    avg_acc = 0.0

    for i, (c, (speaker_indices, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        with torch.no_grad():
            pred, _ = net(c)

        loss, losses = criterion(pred, speaker_indices)
        acc = accuracy(pred, speaker_indices)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()
        
        avg_acc += acc.item()

        print(f'Testing: {i + 1:03d}/{num_repeats:03d} (loss={loss.item() / g.batch_size:.10f}, acc={acc.item() * 100:.4f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= num_repeats * g.batch_size

    avg_acc /= num_repeats * g.batch_size

    logging.debug(f'TEST LOSSES: {avg_losses}')
    logging.debug(f'TEST ACC: {avg_acc * 100}')

    return avg_losses, avg_acc


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
