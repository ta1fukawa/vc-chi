import argparse
import datetime
import logging
import pathlib
import traceback

import matplotlib; matplotlib.use('Agg')
import torch
import torch.utils.tensorboard
import yaml

from modules import common, dataset
from modules import global_value as g
from modules import model
from modules import audio


def main(config_path, model_load_path=None, gpu=0):
    g.code_id = 'apple'
    g.run_id  = datetime.datetime.now().strftime('%Y%m/%d/%H%M%S')

    g.work_dir = pathlib.Path('wd', g.code_id, g.run_id)
    g.work_dir.mkdir(parents=True)

    common.init_logger(g.work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    common.backup_codes(pathlib.Path(__file__).parent, g.work_dir / 'code')

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

    train_dataset = dataset.Dataset(g.use_same_speaker, **g.train_dataset)
    valdt_dataset = dataset.Dataset(g.use_same_speaker, **g.valdt_dataset)
    tests_dataset = dataset.Dataset(g.use_same_speaker, **g.tests_dataset)

    def criterion(c, t, r, q, c_feat, q_feat):
        r_loss = torch.nn.functional.mse_loss(r, t)
        q_loss = torch.nn.functional.mse_loss(q, t)
        code_loss = torch.nn.functional.l1_loss(q_feat, c_feat)
        return r_loss + q_loss + code_loss

    optimizer = torch.optim.Adam(net.parameters(), lr=g.lr)

    (g.work_dir / 'cp').mkdir(parents=True)

    best_train_loss = best_valdt_loss = float('inf')

    with torch.utils.tensorboard.SummaryWriter(g.work_dir / 'tboard') as sw:
        for epoch in range(g.num_epochs):
            train_loss = model_train   (epoch, net, train_dataset, criterion, optimizer)
            valdt_loss = model_validate(epoch, net, valdt_dataset, criterion)

            sw.add_scalars('loss', {'train': train_loss, 'valdt': valdt_loss}, epoch)
            sw.flush()

            logging.info(f'[{epoch:03d}/{g.num_epochs:03d}] TRAIN LOSS: {train_loss:.6f}, VALIDATE LOSS: {valdt_loss:.6f}')

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(net.state_dict(), g.work_dir / 'cp' / 'best_train.pth')

            if valdt_loss < best_valdt_loss:
                best_valdt_loss = valdt_loss
                torch.save(net.state_dict(), g.work_dir / 'cp' / 'best_valdt.pth')

    tests_loss = model_test(net, tests_dataset, criterion)

    logging.info(f'BEST TRAIN LOSS: {best_train_loss:.6f}')
    logging.info(f'BEST VALIDATE LOSS: {best_valdt_loss:.6f}')
    logging.info(f'TEST LOSS: {tests_loss:.6f}')

    predict(net, **g.predict)


def model_train(epoch, net, dataset, criterion, optimizer):
    net.train()

    avg_loss = 0.0

    for i, (c, t, c_emb, s_emb) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)

        c_feat = net.content_enc(c, c_emb)
        r      = net.decoder(c_feat, s_emb)
        q      = r + net.postnet(r)
        q_feat = net.content_enc(q, c_emb)

        loss = criterion(c, t, r, q, c_feat, q_feat)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'[{epoch:03d}/{g.num_epochs:03d}] Training: {i:03d}/{g.train_dataset["num_repeats"]:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')

    avg_loss /= g.train_dataset['num_repeats'] * g.batch_size

    return avg_loss


def model_validate(epoch, net, dataset, criterion):
    net.eval()

    avg_loss = 0.0

    for i, (c, t, c_emb, s_emb) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)

        c_feat = net.content_enc(c, c_emb)
        r      = net.decoder(c_feat, s_emb)
        q      = r + net.postnet(r)
        q_feat = net.content_enc(q, c_emb)

        loss = criterion(c, t, r, q, c_feat, q_feat)
        avg_loss += loss.item()

        print(f'[{epoch:03d}/{g.num_epochs:03d}] Validate: {i:03d}/{g.valdt_dataset["num_repeats"]:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')

    avg_loss /= g.valdt_dataset['num_repeats'] * g.batch_size

    return avg_loss


def model_test(net, dataset, criterion):
    net.eval()

    avg_loss = 0.0

    for i, (c, t, c_emb, s_emb) in enumerate(dataset):
        c = c.to(g.device); t = t.to(g.device)

        c_feat = net.content_enc(c, c_emb)
        r      = net.decoder(c_feat, s_emb)
        q      = r + net.postnet(r)
        q_feat = net.content_enc(q, c_emb)

        loss = criterion(c, t, r, q, c_feat, q_feat)
        avg_loss += loss.item()

        print(f'Testing: {i:03d}/{g.tests_dataset["num_repeats"]:03d} (loss={loss.item() / g.batch_size:.6f})\033[K\033[G', end='')

    avg_loss /= g.tests_dataset['num_repeats'] * g.batch_size

    return avg_loss


def predict(net, source_speaker, target_speaker, speech):
    net.eval()

    c = torch.load(f'{g.mel_dir}/{source_speaker}/{speech}.pt').unsqueeze(0).to(g.device)
    t = torch.load(f'{g.mel_dir}/{target_speaker}/{speech}.pt').unsqueeze(0).to(g.device)
    c_emb = torch.load(f'{g.emb_dir}/{source_speaker}.pt').unsqueeze(0).to(g.device)
    s_emb = torch.load(f'{g.emb_dir}/{target_speaker}.pt').unsqueeze(0).to(g.device)

    c_feat = net.content_enc(c, c_emb)
    r      = net.decoder(c_feat, s_emb)
    q      = r + net.postnet(r)

    angle = torch.load(f'{g.agl_dir}/{target_speaker}/{speech}.pt').unsqueeze(0).to(g.device)

    audio.save_spec_fig(c, t, r, q)
    audio.save_mel_wave(c, t, r, q, angle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',     type=pathlib.Path, default='config.yml')
    parser.add_argument('--model_load_path', type=pathlib.Path)
    parser.add_argument('--gpu',             type=int, default=0)

    try:
        main(**vars(parser.parse_args()))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()
