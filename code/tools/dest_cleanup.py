import pathlib
import shutil

vc_dir = pathlib.Path('./dest/train_vc')

for month_dir in sorted(vc_dir.glob('*')):
    for date_dir in sorted(month_dir.glob('*')):
        for sec_dir in date_dir.glob('*'):
            for dirname in ['wav', 'wave_img', 'mel_img', 'cp', 'tboard', 'mel_data']:
                dir = sec_dir / dirname
                if dir.exists():
                    print(dir)
                    shutil.rmtree(dir)


xvector_dir = pathlib.Path('./dest/train_xvector')

# for date_dir in sorted(xvector_dir.glob('*'))[:-1]:
#     for sec_dir in date_dir.glob('*'):
#         for dirname in ['cp', 'tboard']:
#             dir = sec_dir / dirname
#             if dir.exists():
#                 print(dir)
#                 # shutil.rmtree(dir)
