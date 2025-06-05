#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.util_image import ImageSpliterTh

from pcu.utils import save_range_image
from torchvision import transforms
from models.unet import UNetModelSwin
from datapipe.pcu_dataset import PCUDataSet
from tqdm import tqdm

def upsampling(lq_path):
    lq = np.load(lq_path)  

    lq_torch = torch.tensor(lq, dtype=torch.float32)  

    lq_torch = lq_torch.unsqueeze(0).unsqueeze(0)
    sr_torch = F.interpolate(lq_torch, scale_factor=(4, 1), mode='nearest')

    sr_np = sr_torch.squeeze(0).squeeze(0).numpy()  # 还原 batch 维度

    return sr_np

class ScaleTensor(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    def __call__(self, tensor):
        return tensor*self.scale_factor

class LogTransform(object):
    def __call__(self, tensor):
        return torch.log1p(tensor)

class InverseLogTransform(object):
    def __call__(self, tensor):
        return torch.expm1(tensor)
    
class BaseSampler:
    def __init__(
            self,
            configs,
            sf=4,
            use_amp=True,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            padding_offset=16,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.sf = sf
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend='nccl', init_method='env://')

            
        # if num_gpus > 1:
        #     if mp.get_start_method(allow_none=True) is None:
        #         mp.set_start_method('spawn', force=True)

        #     self.rank = int(os.environ['RANK'])             # 全局 rank
        #     self.local_rank = int(os.environ['LOCAL_RANK']) # 当前节点的 GPU 编号
        #     self.world_size = int(os.environ['WORLD_SIZE']) # 总进程数

        #     torch.cuda.set_device(self.local_rank)          # 正确设定 GPU
        #     dist.init_process_group(backend='nccl', init_method='env://')


        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        params = self.configs.model.get('params', dict)
        model=UNetModelSwin(**params)
        model.cuda()
        # model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            util_net.reload_model(model, ckpt['state_dict'])
        else:
            util_net.reload_model(model, ckpt)
        self.freeze_model(model)
        self.model = model.eval()

        self.autoencoder = None
            

    def load_model_lora(self, model, ckpt_path=None, tag='model'):
        if self.rank == 0:
            self.write_log(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        num_success = 0
        for key, value in model.named_parameters():
            if key in ckpt:
                value.data.copy_(ckpt[key])
                num_success += 1
            else:
                key_parts = key.split('.')
                if 'conv' in key_parts:
                    key_parts.remove('conv')
                new_key = '.'.join(key_parts)
                if new_key in ckpt:
                    value.data.copy_(ckpt[new_key])
                    num_success += 1
        assert num_success == len(ckpt)
        if self.rank == 0:
            self.write_log('Loaded Done')

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class ResShiftSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False, mask=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
            mask: image mask for inpainting
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''

        if noise_repeat:
            self.setup_seed()


        if self.configs.model.params.cond_sr:
            model_kwargs={'sr':y0,}
        else:
            model_kwargs = None

        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                )    # This has included the decoding for latent space


        return results.clamp_(0, 1.0)

    def inference(self, in_path, out_path, mask_path=None, mask_back=True, bs=1, noise_repeat=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
            mask_path: image mask for inpainting
        '''
        def _process_per_image(im_lq_tensor, mask=None):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
                mask: image mask for inpainting, [-1, 1], 1 for unknown area
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''
            context = lambda: torch.amp.autocast('cuda') if self.use_amp else nullcontext()
            # print(im_lq_tensor.shape)
            with context():
                im_sr_tensor = self.sample_func(
                        im_lq_tensor,
                        noise_repeat=noise_repeat,
                        mask=mask,
                        )   
            return im_sr_tensor
        
        def _inference_one(path,out_name):
            with open(path, "rb") as f:
                range_map = np.load(f).astype(np.float32)
            trans = [transforms.ToTensor(), ScaleTensor(1/80),LogTransform()]
            transform = transforms.Compose(trans)
            range_map=transform(range_map)
            range_map=range_map.unsqueeze(0).to("cuda")
            range_map=F.interpolate(range_map, scale_factor=(4, 1), mode='nearest')

            im_sr_tensor = _process_per_image(
                    range_map
                    )
            
            im_sr_tensor=inverse_transform(im_sr_tensor)
            im_sr=im_sr_tensor.squeeze(0).squeeze(0).cpu().numpy()
            im_path = out_path / out_name
            np.save(im_path,im_sr)
        
        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if self.rank == 0:
            import ipdb;ipdb.set_trace()
            assert in_path.exists()
            if not out_path.exists():
                out_path.mkdir(parents=True)

        if self.num_gpus > 1:
            dist.barrier()
        inverse_trans=[InverseLogTransform(),ScaleTensor(80)]
        inverse_transform=transforms.Compose(inverse_trans)
        if in_path.is_dir():
            # TO BE MODIFIED
            dataset = PCUDataSet(self.configs.data.val)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            for data in tqdm(dataloader,desc="infer..."):
                micro_batchsize = math.ceil(bs / self.num_gpus)
                ind_start = self.rank * micro_batchsize
                ind_end = ind_start + micro_batchsize
                micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}
                # import ipdb
                # ipdb.set_trace()
                if micro_data['sr'].shape[0] > 0:
                    results = _process_per_image(
                            micro_data['sr'].cuda()
                            )    # b x h x w x c, [0, 1], RGB
                    results=inverse_transform(results)
                    results=results.squeeze(1).cpu().numpy()
                    for jj in range(results.shape[0]):
                        # cur_res=results[jj]
                        # import ipdb
                        # ipdb.set_trace()
                        # cur_res=inverse_transform(cur_res)
                        # cur_res=cur_res.squeeze(0).cpu().numpy()
                        name=micro_data['sr_path'][jj].split("/")[-1]
                        im_path = out_path / f"{name}"
                        np.save(im_path,results[jj])
            if self.num_gpus > 1:
                dist.barrier()

            # root=in_path
            # files=os.listdir(root)
            # for filename in tqdm(files):
            #     filename="00001830.npy"
            #     range_path=os.path.join(root,filename)
            #     _inference_one(range_path,filename)
        else:
            _inference_one(in_path,f"{in_path.stem}_sr.npy")

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")
    
   
        

if __name__ == '__main__':
    pass

# # TO BE MODIFIED
# dataset = PCUDataSet(self.configs.data.val)
# self.write_log(f'Find {len(dataset)} images in {in_path}')
# dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=bs,
#         shuffle=False,
#         drop_last=False,
#         )
# for data in tqdm(dataloader,desc="infer..."):
#     micro_batchsize = math.ceil(bs / self.num_gpus)
#     ind_start = self.rank * micro_batchsize
#     ind_end = ind_start + micro_batchsize
#     micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}
#     if micro_data['sr'].shape[0] > 0:
#         results = _process_per_image(
#                 micro_data['sr'].cuda()
#                 )    # b x h x w x c, [0, 1], RGB
#         # results=inverse_transform(results)
#         # results=results.squeeze(1).cpu().numpy()
#         for jj in range(results.shape[0]):
#             cur_res=results[jj]
#             import ipdb
#             ipdb.set_trace()
#             cur_res=inverse_transform(cur_res)
#             cur_res=cur_res.squeeze(0).cpu().numpy()
#             name=micro_data['sr_path'][jj].split("/")[-1]
#             im_path = out_path / f"{name}"
#             np.save(im_path,results[jj])
# if self.num_gpus > 1:
#     dist.barrier()

