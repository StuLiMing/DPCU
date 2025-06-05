import os
from torch.utils import data as data
from torchvision import transforms


import torch
import numpy as np


class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix

        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='basicsr'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')



class ScaleTensor(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    def __call__(self, tensor):
        return tensor*self.scale_factor

class LogTransform(object):
    def __call__(self, tensor):
        return torch.log1p(tensor)

hflip = transforms.RandomHorizontalFlip()

def transform_augment(imgs):    
    imgs = torch.stack(imgs, 0)
    imgs = hflip(imgs)
    imgs = torch.unbind(imgs, dim=0)

@DATASET_REGISTRY.register(suffix='pcu')
class PCUDataSet(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PCUDataSet, self).__init__()
        self.opt = opt

        self.gt_folder, self.lq_folder, self.sr_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt["dataroot_sr"]
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        assert self.opt['meta_info'] is not None
        with open(self.opt['meta_info']) as fin:
            paths = [line.strip() for line in fin]
        self.paths = []
        for path in paths:
            gt_path, lq_path,sr_path = path.split(',')
            gt_path = os.path.join(self.gt_folder, gt_path)
            lq_path = os.path.join(self.lq_folder, lq_path)
            sr_path = os.path.join(self.sr_folder, sr_path)
            self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path), ('sr_path', sr_path)]))

    def __getitem__(self, index):
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        sr_path = self.paths[index]['sr_path']

        img_lq=np.load(lq_path).astype(np.float32)
        img_gt=np.load(gt_path).astype(np.float32)
        img_sr=np.load(sr_path).astype(np.float32)

        trans = [transforms.ToTensor(), ScaleTensor(1/80),LogTransform()]
        transform = transforms.Compose(trans)

        img_lq=transform(img_lq)
        img_gt=transform(img_gt)
        img_sr=transform(img_sr)

        # if torch.rand(1) > 0.5:  # 以0.5的概率执行翻转
        #     img_lq=torch.flip(img_lq, dims=[-1])
        #     img_gt=torch.flip(img_gt, dims=[-1])
        #     img_sr=torch.flip(img_gt, dims=[-1])

        # CHW
        # C=1
        # Tensor
        # cpu
        # float
        # 0-1

        return {'lq': img_lq, 'gt': img_gt, 'sr': img_sr, 'lq_path': lq_path, 'gt_path': gt_path, 'sr_path': sr_path}

    def __len__(self):
        return len(self.paths)
