path="/amax/lm/Datahouse/KITTI/train_16_64/hr_64/00000000.npy"
path_l="/amax/lm/Datahouse/KITTI/train_16_64/lr_16/00000000.npy"


import numpy as np

a=np.load(path)
b=np.load(path_l)


# hr:(64*1024)
# lr:(16*1024)
import ipdb
ipdb.set_trace()