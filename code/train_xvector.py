import argparse
import csv
import logging
import pathlib
import traceback

import torch
import torch.utils.tensorboard

from modules import audio, common, dataset
from modules import global_value as g
from modules import xvector


def main(config_path, note):
    common.custom_init(config_path, '%Y%m%d/%H%M%S', note)

    net = xvector.Net().to(g.device)
    logging.debug(f'MODEL: {net}')

    if g.model_load_path is not None:
        net.load_state_dict(torch.load(g.model_load_path))
        logging.debug(f'LOAD MODEL: {g.model_load_path}')

    nll_criterion = torch.nn.NLLLoss()
    def criterion(pred, emb, indices):
        nll_loss = nll_criterion(pred, indices)

        cos_sim_loss = 0.0
        for i in range(indices.shape[0]):
            for j in range(indices.shape[0]):
                if i == j:
                    continue
                elif indices[i] == indices[j]:
                    cos_sim_loss += 1 - torch.nn.functional.cosine_similarity(emb[i], emb[j], dim=0)
                else:
                    cos_sim_loss += 1 + torch.nn.functional.cosine_similarity(emb[i], emb[j], dim=0)
        cos_sim_loss /= indices.shape[0] * (indices.shape[0] - 1) * 2

        acc = (pred.argmax(dim=1) == indices).float().mean()

        loss = nll_loss

        losses = {
            'nll_loss': nll_loss,
            'cos_sim_loss': cos_sim_loss,
            'loss': loss,
            'acc': acc,
        }

        return loss, losses

    (g.work_dir / 'cp').mkdir(parents=True)

    with torch.utils.tensorboard.SummaryWriter(g.work_dir / 'tboard') as sw:
        total_epoch = 0

        for stage_no, stage in enumerate(g.stages):
            logging.info(f'STAGE: {stage}')
            common.update_note_status(f'stage_{stage_no}')

            net.set_cassifier(stage['speaker_size'])
            logging.debug(f'MODEL: {net}')

            if stage['only_classifier']:
                parameters = net.classifier.parameters()
            else:
                parameters = net.parameters()

            if stage['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(parameters, lr=stage['lr'])
            elif stage['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(parameters, lr=stage['lr'], momentum=stage['momentum'])
            logging.debug(f'SET OPTIMIZER: {optimizer}')

            train_dataset = dataset.PnmDataset(stage['speaker_size'], **g.train_dataset)
            valdt_dataset = dataset.PnmDataset(stage['speaker_size'], **g.valdt_dataset)
            tests_dataset = dataset.PnmDataset(stage['speaker_size'], **g.tests_dataset)

            patience = 0

            best_train_loss = best_valdt_loss = {'loss': float('inf')}

            for epoch in range(stage['num_epochs']):
                logging.info(f'EPOCH: {epoch + 1} (TOTAL: {total_epoch + 1})')

                train_loss = model_train   (net, train_dataset, criterion, optimizer)
                valdt_loss = model_validate(net, valdt_dataset, criterion)

                logging.info(f'TRAIN LOSS: {train_loss["loss"]:.10f}, VALDT LOSS: {valdt_loss["loss"]:.10f}')
                logging.info(f'TRAIN ACC: {train_loss["acc"]:.4f}, VALDT ACC: {valdt_loss["acc"]:.4f}')

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

            tests_loss = model_test(net, tests_dataset, criterion)

            logging.info(f'BEST TRAIN LOSS: {best_train_loss["loss"]:.10f}, BEST VALDT LOSS: {best_valdt_loss["loss"]:.10f}, TEST LOSS: {tests_loss["loss"]:.10f}')
            logging.info(f'BEST TRAIN ACC: {best_train_loss["acc"]:.4f}, BEST VALDT ACC: {best_valdt_loss["acc"]:.4f}, TEST ACC: {tests_loss["acc"]:.4f}')

    predict(net)


def model_train(net, dataset, criterion, optimizer):
    net.train()

    avg_losses = {}

    for i, (c, (speaker_indices, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        pred, emb = net(c)

        loss, losses = criterion(pred, emb, speaker_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Training: {i + 1:03d}/{g.train_dataset["num_repeats"]:03d} (loss={loss.item():.10f}, acc={losses["acc"].item():.4f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.train_dataset['num_repeats']

    logging.debug(f'TRAIN LOSSES: {avg_losses}')
    logging.debug(f'TRAIN ACC: {avg_losses["acc"]:.4f}')

    return avg_losses


def model_validate(net, dataset, criterion):
    net.eval()

    avg_losses = {}

    dataset.set_seed(0)
    for i, (c, (speaker_indices, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        with torch.no_grad():
            pred, emb = net(c)

        loss, losses = criterion(pred, emb, speaker_indices)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Validate: {i + 1:03d}/{g.valdt_dataset["num_repeats"]:03d} (loss={loss.item():.10f}, acc={losses["acc"].item():.4f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.valdt_dataset['num_repeats']

    logging.debug(f'VALIDATE LOSSES: {avg_losses}')

    return avg_losses


def model_test(net, dataset, criterion):
    net.eval()

    avg_losses = {}

    dataset.set_seed(0)
    for i, (c, (speaker_indices, _)) in enumerate(dataset):
        c = c.to(g.device)
        speaker_indices = speaker_indices.to(g.device)

        with torch.no_grad():
            pred, emb = net(c)

        loss, losses = criterion(pred, emb, speaker_indices)

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0.0) + v.item()

        print(f'Testing: {i + 1:03d}/{g.tests_dataset["num_repeats"]:03d} (loss={loss.item():.10f}, acc={losses["acc"].item():.4f})\033[K\033[G', end='')

    print('\033[K\033[G', end='')

    for k, v in avg_losses.items():
        avg_losses[k] /= g.tests_dataset['num_repeats']

    logging.debug(f'TEST LOSSES: {avg_losses}')
    logging.debug(f'TEST ACC: {avg_losses["acc"]:.4f}')

    return avg_losses


def predict(net):
    net.eval()

    wav_dir = pathlib.Path(g.all_dir)
    emb_dir = pathlib.Path(g.work_dir, 'emb')

    emb_dir.mkdir(parents=True, exist_ok=True)

    embs_left = []; embs_right = []
    for speaker in sorted(wav_dir.iterdir()):
        if not speaker.is_dir():
            continue

        speaker_waves = dataset.load_speaker_waves(speaker)
        speaker_mels = [
            dataset.padding(torch.from_numpy(audio.fast_stft(wave).T[1:]), g.pad_pnm_len)
            for wave in speaker_waves
        ]

        speaker_mels = torch.stack(speaker_mels, dim=0).to(g.device)
        speaker_embs = []
        for i in range((speaker_mels.shape[0] - 1) // g.batch_size + 1):
            with torch.no_grad():
                _, speaker_emb = net(speaker_mels[i * g.batch_size: (i + 1) * g.batch_size])
                speaker_embs.append(speaker_emb)
        speaker_embs = torch.cat(speaker_embs, dim=0)
        speaker_emb = torch.mean(speaker_embs, dim=0).cpu()
        torch.save(speaker_emb, str(emb_dir / f'{speaker.name}.pt'))

        emb_left  = torch.mean(speaker_embs[:speaker_embs.shape[0] // 2], dim=0).cpu()
        emb_right = torch.mean(speaker_embs[speaker_embs.shape[0] // 2:], dim=0).cpu()

        embs_left.append(emb_left)
        embs_right.append(emb_right)

    cos_sim_mat = torch.empty((len(embs_left), len(embs_right)))
    vec_dis_mat = torch.empty((len(embs_left), len(embs_right)))
    for i, emb_i in enumerate(embs_left):
        for j, emb_j in enumerate(embs_right):
            cos_sim_mat[i, j] = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=0).item()
            vec_dis_mat[i, j] = torch.norm(emb_i - emb_j, p=2).item()

    with open(g.work_dir / 'emb_cossim.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(cos_sim_mat.numpy())

    with open(g.work_dir / 'emb_dffdis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(vec_dis_mat.numpy())

    nodiag_cos_sim = torch.tril(cos_sim_mat, -1)[:, :-1] + torch.triu(cos_sim_mat, 1)[:, 1:]
    nodiag_vec_dis = torch.tril(vec_dis_mat, -1)[:, :-1] + torch.triu(vec_dis_mat, 1)[:, 1:]

    logging.info(f'COS SIM: {torch.mean(nodiag_cos_sim):.6f} (STD: {torch.std(nodiag_cos_sim):.6f})')
    logging.info(f'VEC DISTANCE: {torch.mean(nodiag_vec_dis):.6f} (STD: {torch.std(nodiag_vec_dis):.6f})')
    logging.info(f'COS SIM/DIAG: {torch.mean(torch.diag(cos_sim_mat, 0)):.6f} (STD: {torch.std(torch.diag(cos_sim_mat, 0)):.6f})')
    logging.info(f'VEC DISTANCE/DIAG: {torch.mean(torch.diag(vec_dis_mat, 0)):.6f} (STD: {torch.std(torch.diag(vec_dis_mat, 0)):.6f})')

    return cos_sim_mat, vec_dis_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=pathlib.Path, default='./configs/xvector_config.yml')
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
