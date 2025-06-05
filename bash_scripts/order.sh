train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py --cfg_path configs/train.yaml --save_dir save_train

train_dbg
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py --cfg_path configs/train_dbg.yaml --save_dir save_train_dbg


4gpu_train
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --cfg_path configs/train.yaml --save_dir save_train


infer_one
CUDA_VISIBLE_DEVICES=4 python infer.py -i test_pcu/00001830.npy -o test_pcu --scale 4 --bs 1

infer_batch_naive
CUDA_VISIBLE_DEVICES=4 python infer.py -i /amax/lm/Datahouse/KITTI/val_16_64/lr_16 -o test_batch_out --scale 4 --bs 1


# don't work
infer_batch
CUDA_VISIBLE_DEVICES=4,5,6,7  torchrun --standalone --nproc_per_node=4 --nnodes=1 infer.py -i test_batch -o test_batch_out --scale 4 --bs 8

infer_batch_dbg
CUDA_VISIBLE_DEVICES=4  torchrun --standalone --nproc_per_node=1 --nnodes=1 infer.py -i /amax/lm/Datahouse/KITTI/val_16_64/lr_16 -o test_batch_out --scale 4 --bs 2
