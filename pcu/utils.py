import numpy as np
import matplotlib.pyplot as plt
from .trans23 import rang2pc_kitti

def save_pc(pc:str|np.ndarray,save_path:str):
    if isinstance(pc,str):
        if pc.endswith(".bin"):
            lidar_data = np.fromfile(pc, dtype=np.float32).reshape(-1, 4)

    if isinstance(pc,np.ndarray):
        lidar_data=pc[...,:3]
    
    if save_path.endswith(".ply"):
        header = f"""ply\nformat ascii 1.0\nelement vertex {lidar_data.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"""
        # 写入文件
        with open(save_path, 'w') as f:
            f.write(header)
            for point in lidar_data:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def read_npy(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        range_map = np.load(f)
    return range_map.astype(np.float32)

def read_ply(ply_path):
    with open(ply_path,"r",encoding="utf-8") as f:
        pc_str_lst=f.read().split(sep="\n")[7:]
        pc_array_lst=[]
        for pc_line in pc_str_lst:
            if pc_line=="":
                continue
            pc_line=list(map(float,pc_line.split()))
            pc_array_lst.append(pc_line)
        return np.array(pc_array_lst,dtype=np.float32)

def vis_range_image(range_image,bg=True):
    # 可视化图像

    if bg:
        plt.imshow(range_image,cmap="jet")
        plt.axis('off')
    else:
        dpi = 100  # 每英寸点数，可以根据需要调整
        height, width = range_image.shape
        plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        # 显示图像
        plt.imshow(range_image, cmap="jet", aspect='auto')

        # 去除坐标轴
        plt.axis('off')

        # 调整子图参数，使图像填满整个图形区域
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.margins(0, 0)
    plt.show()

def save_range_image(range_image, img_path):
    """
    保存范围图像，不包含白色背景，仅保存图像本身。

    参数:
    - range_image: 要保存的图像数据（通常是二维数组）。
    - img_path: 保存图像的路径（包括文件名和扩展名，如 'output.png'）。
    """
    # 创建一个新的图形，设置大小为图像的像素大小除以 DPI
    dpi = 100  # 每英寸点数，可以根据需要调整
    height, width = range_image.shape
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # 显示图像
    plt.imshow(range_image, cmap="jet", aspect='auto')

    # 去除坐标轴
    plt.axis('off')

    # 调整子图参数，使图像填满整个图形区域
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0, 0)

    # 保存图像
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)

    # 关闭图形以释放内存
    plt.close()

if __name__=="__main__":
    # range_path="./testdata/00000000_2/00000000.npy"
    # range_img=read_npy(range_path)
    # pc=rang2pc_kitti(range_img,deviation=None)
    # # save_pc(pc,"./testdata/00000000_2/00000000_2.ply")
    # save_range_image(range_img,"./testdata/00000000_2/00000000.png")



    # 构造一个简单的渐变图像：从左到右值逐渐变大
    test_img = np.tile(np.linspace(0, 1, 256), (50, 1))  # shape: (50, 256)

    # 保存使用 jet 色图的图像
    save_range_image(test_img, "range_jet.png")

