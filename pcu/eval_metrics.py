import sys
from chamfer_distance import ChamferDistance as chamfer_dist

import torch
import numpy as np

from .utils import read_npy
from .trans23 import rang2pc_kitti

def voxelize_point_cloud(point_cloud, grid_size, min_coord, max_coord):
    # import ipdb
    # ipdb.set_trace()
    # Calculate the dimensions of the voxel grid
    dimensions = ((max_coord - min_coord) / grid_size).astype(np.int32) + 1
    # Create the voxel grid
    voxel_grid = np.zeros(dimensions)

    # Assign points to voxels
    indices = ((point_cloud - min_coord) / grid_size).astype(np.int32)
    voxel_grid[tuple(indices.T)] = True

    return voxel_grid

def calculate_IoU(pcd_pred,pcd_gt):

    grid_size=0.1

    pcd_all = np.vstack((pcd_pred, pcd_gt))
    min_coord = np.min(pcd_all, axis=0)
    max_coord = np.max(pcd_all, axis=0)
    # import ipdb
    # ipdb.set_trace()
    # Voxelize the ground truth and prediction point clouds
    voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
    voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)
    intersection = np.logical_and(voxel_grid_predicted, voxel_grid_ground_truth)
    union = np.logical_or(voxel_grid_predicted, voxel_grid_ground_truth)
    iou = np.sum(intersection) / np.sum(union)
    return iou
    # return iou

def chamfer_distance(points1, points2, num_points = None):
    source = torch.from_numpy(points1[None, :]).cuda()
    target = torch.from_numpy(points2[None, :]).cuda()


    chd = chamfer_dist()
    dist1, dist2, _, _ = chd(source, target)
    cdist = (torch.mean(dist1)) + (torch.mean(dist2)) if num_points is None else (dist1.sum()/num_points) + (dist2.sum()/num_points)

    return cdist.detach().cpu()

def calculate_mae(pred_img:torch.Tensor,gt_img:torch.Tensor):
    if isinstance(pred_img,torch.Tensor) and isinstance(gt_img,torch.Tensor):
        loss_map = (pred_img -gt_img).abs()
        pixel_loss_one_input = loss_map.mean()
    elif isinstance(pred_img,np.ndarray) and isinstance(gt_img,np.ndarray):
        loss_map=np.abs(pred_img -gt_img)
        pixel_loss_one_input = loss_map.mean()
    return pixel_loss_one_input

if __name__=="__main__":    
    # gt_img=read_npy("test_pcu_00000000/hr.npy")
    # pred_img=read_npy("./test_batch_out/00000000.npy")
    # # pred_img=read_npy("test_pcu_00000000/lr_sr.npy")
    
    # mae=calculate_mae(pred_img,gt_img)
    
    # print(f"mae:{mae}")

    import os
    from tqdm import tqdm

    gt_root="/amax/lm/Datahouse/KITTI/val_16_64/hr_64"
    pred_root="test_batch_out"
    # pred_root="testdata/all"

    gt_lst=os.listdir(gt_root)
    pred_lst=os.listdir(pred_root)
    gt_lst.sort()
    pred_lst.sort()
    mae_lst=[]
    cd_lst=[]
    iou_lst=[]
    for i in tqdm(range(len(gt_lst))):
        
        gt_img=read_npy(os.path.join(gt_root,gt_lst[i]))
        pred_img=read_npy(os.path.join(pred_root,pred_lst[i]))
        # mask=np.zeros_like(gt_img,dtype=np.bool)
        # mask[range(0,64,4),:]=True
        # pred_img=pred_img[range(0,64,4),:]
        # gt_img=gt_img[range(0,64,4),:]
        # pred_img=np.log(pred_img/80+1)*80
        # gt_img=np.log(gt_img/80+1)*80
        # pred_img = np.where((pred_img >= 0) & (pred_img <= 80), pred_img, 0)

        pred_img=np.clip(pred_img, 0, 80)
        pred_img[range(0,64,4),:]=gt_img[range(0,64,4),:]
        mae=calculate_mae(pred_img,gt_img)
        mae_lst.append(mae)
        pred_pcd=rang2pc_kitti(pred_img,deviation=None)
        gt_pcd=rang2pc_kitti(gt_img,deviation=None)
        cd=chamfer_distance(pred_pcd,gt_pcd)
        cd_lst.append(cd)   

        iou=calculate_IoU(pred_pcd,gt_pcd)
        iou_lst.append(iou)
        # import ipdb
        # ipdb.set_trace()
        # print(f"mae:{mae}")

    mae_lst=np.array(mae_lst)
    cd_lst=np.array(cd_lst)
    iou_lst=np.array(iou_lst)
    print(f"mae:{mae_lst.mean()}")
    print(f"cd:{cd_lst.mean()}")
    print(f"iou:{iou_lst.mean()}")
    import ipdb;ipdb.set_trace()

# mae       iou         cd
# 0.5929    0.3527      0.3441

# no real fill: 0.6026
# nothing:  0.5930
# clip:     0.5929  best
# where:    0.5956