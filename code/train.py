import argparse
import datetime
import logging
import pathlib
import traceback

import numpy as np
import torch
import torch.utils.tensorboard
import yaml

from modules import common
from modules import global_value as g
from modules import model
from modules import dataset


def main(config_path, model_load_path=None, gpu=0):
    g.code_id = 'apple'
    g.run_id  = datetime.datetime.now().strftime('%Y%m/%d/%H%M%S')

    work_dir = pathlib.Path('wd', g.code_id, g.run_id)
    work_dir.mkdir(parents=True)

    common.init_logger(work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    common.backup_codes(pathlib.Path(__file__).parent, work_dir / 'code')

    config = yaml.load(config_path.open(mode='r'), Loader=yaml.FullLoader)
    logging.info(f'CONFIG: {config}')

    for k, v in config.items():
        setattr(g, k, v)

    if gpu >= 0:
        assert torch.cuda.is_available()
        g.device = torch.device(f'cuda:{gpu}')
    else:
        g.device = torch.device('cpu')

    net = model.Net().to(g.device)
    
    if model_load_path is not None:
        net.load_state_dict(torch.load(model_load_path, map_location=g.device))
        logging.info(f'LOAD MODEL: {model_load_path}')

    train_dataset = dataset.Dataset(g.use_same_speaker, test_mode=False)
    test_dataset  = dataset.Dataset(g.use_same_speaker, test_mode=True)

    def criterion(c, s, t, r, q, c_feat, q_feat):
        r_loss = torch.nn.functional.mse_loss(r, t)
        q_loss = torch.nn.functional.mse_loss(q, t)
        code_loss = torch.nn.functional.l1_loss(q_feat, c_feat)
        return r_loss + q_loss + code_loss

    optimizer = torch.optim.Adam(net.parameters(), lr=g.lr)

    (work_dir / 'cp').mkdir(parents=True)

    best_train_loss = best_test_loss = float('inf')

    with torch.utils.tensorboard.SummaryWriter(work_dir / 'tboard') as sw:
        for epoch in range(g.num_epochs):
            train_loss = train(epoch, net, train_dataset, criterion, optimizer)
            test_loss  = test (epoch, net, test_dataset,  criterion)

            sw.add_scalars('loss', {'train': train_loss, 'test': test_loss}, epoch)
            sw.flush()

            logging.info(f'[{epoch:03d}/{g.num_epochs:03d}] TRAIN LOSS: {train_loss:.6f}, TEST LOSS: {test_loss:.6f}')

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(net.state_dict(), work_dir / 'cp' / 'best_train.pth')

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(net.state_dict(), work_dir / 'cp' / 'best_test.pth')

    logging.info(f'BEST TRAIN LOSS: {best_train_loss:.6f}')
    logging.info(f'BEST TEST LOSS: {best_test_loss:.6f}')

def train(epoch, net, dataset, criterion, optimizer):
    net.train()

    avg_loss = 0.0

    for i, (c, s, t) in enumerate(dataset):
        c = c.to(g.device); s = s.to(g.device); t = t.to(g.device)

        c_emb  = net.style_enc(c)
        s_emb  = net.style_enc(s)
        c_feat = net.content_enc(c, c_emb)
        r      = net.decoder(c_feat, s_emb)
        q      = r + net.postnet(r)
        q_feat = net.content_enc(q, c_emb)

        loss = criterion(c, s, t, r, q, c_feat, q_feat)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'[{epoch:03d}/{g.num_epochs:03d}] Training: {i:03d}/{g.num_repeats:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')

    avg_loss /= g.num_repeats * g.batch_size

    return avg_loss


def test(epoch, net, dataset, criterion):
    net.eval()

    avg_loss = 0.0

    for i, (c, s, t) in enumerate(dataset):
        c = c.to(g.device); s = s.to(g.device); t = t.to(g.device)

        c_emb  = net.style_enc(c)
        s_emb  = net.style_enc(s)
        c_feat = net.content_enc(c, c_emb)
        r      = net.decoder(c_feat, s_emb)
        q      = r + net.postnet(r)
        q_feat = net.content_enc(q, c_emb)

        loss = criterion(c, s, t, r, q, c_feat, q_feat)
        avg_loss += loss.item()

        print(f'[{epoch:03d}/{g.num_epochs:03d}] Testing: {i:03d}/{g.num_repeats:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')

    avg_loss /= g.num_test_repeats * g.batch_size

    return avg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',     type=pathlib.Path, default='config.yml')
    parser.add_argument('--model_load_path', type=pathlib.Path)
    parser.add_argument('--gpu',             type=int, default=0)

    args = [
        '--gpu', '1',
    ]

    try:
        main(**vars(parser.parse_args(args)))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()
