import argparse
import datetime
import logging
import pathlib
from pickletools import optimize
import traceback

import numpy as np
import torch
import torch.utils.tensorboard
import yaml

from modules import common
from modules import global_value as g
from modules import model
from modules import dataset

def main(config_path):
    assert torch.cuda.is_available()

    g.code_id = 'apple'
    g.run_id  = datetime.datetime.now().strftime('%Y%m/%d/%H%M%S')

    g.device = torch.device('cuda:0')

    work_dir = pathlib.Path('wd', g.code_id, g.run_id)
    work_dir.mkdir(parents=True)

    common.init_logger(work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    common.backup_codes(pathlib.Path(__file__).parent, work_dir / 'code')

    config = yaml.load(config_path.open(mode='r'), Loader=yaml.FullLoader)
    logging.info(f'CONFIG: {config}')

    for k, v in config.items():
        setattr(g, k, v)

    net = model.Net().to(g.device)
    
    if g.load_model_path is not None:
        net.load_state_dict(torch.load(g.load_model_path, map_location=g.device))
        logging.info(f'LOAD MODEL: {g.load_model_path}')

    train_dataset = dataset.Dataset()
    test_dataset  = dataset.Dataset()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=g.lr)

    (work_dir / 'cp').mkdir(parents=True)

    best_train_loss = 1e+10
    best_test_loss  = 1e+10

    for epoch in range(g.epochs):
        train_loss = train(net, train_dataset, criterion, optimizer)
        test_loss  = test(net, test_dataset, criterion)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(net.state_dict(), work_dir / 'cp' / 'best_test.pth')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(net.state_dict(), work_dir / 'cp' / 'best_train.pth')

        logging.info(f'EPOCH: {epoch:03d}/{g.epochs:03d}')
        logging.info(f'TRAIN: {train_loss:.6f}')
        logging.info(f'TEST : {test_loss :.6f}')

def train(net, train_dataset, criterion, optimizer):
    net.train()

    train_loss = 0.0

    for i, (c, s) in enumerate(train_dataset):
        c = c.to(g.device)
        s = s.to(g.device)

        c_emb = net.style_enc(c)
        s_emb = net.style_enc(s)
        code = net.content_enc(c, c_emb)
        r = net.decoder(code, s_emb)
        q = net.postnet(r)

        loss = criterion(q, c)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'TRAIN: [{i:03d}/{len(train_dataset):03d}] loss={loss.item():.6f}\033[K\033[G', end='')

    train_loss /= len(train_dataset)

    return train_loss

def test(net, test_dataset, criterion):
    net.eval()

    test_loss = 0.0

    for i, (c, s) in enumerate(test_dataset):
        c = c.to(g.device)
        s = s.to(g.device)

        c_emb = net.style_enc(c)
        s_emb = net.style_enc(s)
        code = net.content_enc(c, c_emb)
        r = net.decoder(code, s_emb)
        q = net.postnet(r)

        loss = criterion(q, c)
        test_loss += loss.item()

        print(f'TEST: [{i:03d}/{len(test_dataset):03d}] loss={loss.item():.6f}\033[K\033[G', end='')

    test_loss /= len(test_dataset)

    return test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=pathlib.Path)

    args = [
        'config.yaml'
    ]

    try:
        main(**vars(parser.parse_args(args)))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        # おまじない
        gc.collect()
        torch.cuda.empty_cache()