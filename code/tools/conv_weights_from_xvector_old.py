import sys

sys.path.append('code')

from modules import global_value as g
from modules import xvector_old, xvector

xvector_old.FullModel(2, n_freq=1024, nclasses=80)

