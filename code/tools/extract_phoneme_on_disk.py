import pathlib
import numpy as np
import torch

src_path = pathlib.Path('./dataset/seiren_jvs011/pnm_spc_all')
dst_path = pathlib.Path('/tmp/pnm_spc_all')
dst_path.mkdir(parents=True, exist_ok=True)

for speaker in sorted(src_path.iterdir()):
    speaker_pnm = np.load(speaker, allow_pickle=True)

    pnm_dir = dst_path / speaker.name
    pnm_dir.mkdir(exist_ok=True)

    for i, pnm in enumerate(speaker_pnm):
        torch.save(torch.from_numpy(pnm), pnm_dir / f'{i:06d}.pt')

    print(f'{speaker.name} done')
