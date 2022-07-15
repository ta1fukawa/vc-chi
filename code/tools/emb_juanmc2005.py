import csv
import pathlib

import torch

model = torch.hub.load('pyannote/pyannote-audio', 'emb')

work_dir = pathlib.Path('./dest/_archive/emb_juanmc2005')
wav_dir = pathlib.Path('./dataset/jvs_ver1/wav_all')
emb_dir = work_dir / 'emb'

emb_dir.mkdir(parents=True, exist_ok=True)

embs = []
embs_left = []; embs_right = []
for speaker in sorted(wav_dir.iterdir()):
    if not speaker.is_dir():
        continue

    speaker_embs = []
    for speech in sorted(speaker.iterdir()):
        speaker_emb = model({'audio': speech})
        speaker_embs.append(torch.from_numpy(speaker_emb.data))
    speaker_embs = torch.cat(speaker_embs, dim=0)
    speaker_emb = torch.mean(speaker_embs, dim=0).cpu()
    torch.save(speaker_emb, str(emb_dir / f'{speaker.name}.pt'))

    embs.append(speaker_emb)

    emb_left  = torch.mean(speaker_embs[:speaker_embs.shape[0] // 2], dim=0).cpu()
    emb_right = torch.mean(speaker_embs[speaker_embs.shape[0] // 2:], dim=0).cpu()

    embs_left.append(emb_left)
    embs_right.append(emb_right)

embs = torch.stack(embs, dim=0)
torch.save(embs, work_dir / 'embs.pt')

cos_sim_mat = torch.empty((len(embs_left), len(embs_right)))
vec_dis_mat = torch.empty((len(embs_left), len(embs_right)))
for i, emb_i in enumerate(embs_left):
    for j, emb_j in enumerate(embs_right):
        cos_sim_mat[i, j] = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=0).item()
        vec_dis_mat[i, j] = torch.norm(emb_i - emb_j, p=2).item()

with open(work_dir / 'emb_cossim.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(cos_sim_mat.numpy())

with open(work_dir / 'emb_dffdis.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(vec_dis_mat.numpy())

nodiag_cos_sim = torch.tril(cos_sim_mat, -1)[:, :-1] + torch.triu(cos_sim_mat, 1)[:, 1:]
nodiag_vec_dis = torch.tril(vec_dis_mat, -1)[:, :-1] + torch.triu(vec_dis_mat, 1)[:, 1:]

print(f'COS SIM: {torch.mean(nodiag_cos_sim):.6f} (STD: {torch.std(nodiag_cos_sim):.6f})')
print(f'VEC DISTANCE: {torch.mean(nodiag_vec_dis):.6f} (STD: {torch.std(nodiag_vec_dis):.6f})')
print(f'COS SIM/DIAG: {torch.mean(torch.diag(cos_sim_mat, 0)):.6f} (STD: {torch.std(torch.diag(cos_sim_mat, 0)):.6f})')
print(f'VEC DISTANCE/DIAG: {torch.mean(torch.diag(vec_dis_mat, 0)):.6f} (STD: {torch.std(torch.diag(vec_dis_mat, 0)):.6f})')
