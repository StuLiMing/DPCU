import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
import os



def upsampling(lq_path,range_img):
    lq = np.load(lq_path)  

    lq_torch = torch.tensor(lq, dtype=torch.float32)  

    lq_torch = lq_torch.unsqueeze(0).unsqueeze(0)
    sr_torch = F.interpolate(lq_torch, scale_factor=(4, 1), mode='nearest')

    sr_np = sr_torch.squeeze(0).squeeze(0).numpy()  # 还原 batch 维度

    sr_path=os.path.join("/amax/lm/Datahouse/KITTI/val_16_64/sr_16_64",range_img)
    np.save(sr_path, sr_np)

lr_root="/amax/lm/Datahouse/KITTI/val_16_64/lr_16"
range_img_lst=os.listdir(lr_root)
for range_img in tqdm(range_img_lst):
    lq_path=os.path.join(lr_root,range_img)
    upsampling(lq_path,range_img)


