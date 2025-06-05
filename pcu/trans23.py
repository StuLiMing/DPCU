import numpy as np
from scipy import interpolate

def interpolate_range_image(range_image):
# 创建零点的掩码
    zero_mask = (range_image == 0)

    up = np.roll(range_image, -1, axis=0)
    up[-1, :] = 1  # 假设边界外的点不为0，可以根据实际情况调整

    down = np.roll(range_image, 1, axis=0)
    down[0, :] = 1  # 假设边界外的点不为0

    left = np.roll(range_image, -1, axis=1)
    left[:, -1] = 1  # 假设边界外的点不为0

    right = np.roll(range_image, 1, axis=1)
    right[:, 0] = 1  # 假设边界外的点不为0

    # 创建一个新的掩码，只有当当前点为0且上下左右四个相邻点都不为0时为True
    zero_mask = zero_mask & ((left != 0) & (right != 0) & (up!=0) & (down!=0))

    # 获取有效（非零）点的坐标和对应的值
    y_indices, x_indices = np.nonzero(~zero_mask)
    points = np.vstack((x_indices, y_indices)).T
    values = range_image[~zero_mask]

    # 获取需要插值的点的坐标
    missing_y, missing_x = np.nonzero(zero_mask)
    missing_points = np.vstack((missing_x, missing_y)).T

    # 执行插值
    # method 可以选择 'nearest', 'linear', 'cubic' 等，根据需要选择
    interpolated_values = interpolate.griddata(points, values, missing_points, method='linear')

    # 如果插值结果中有 NaN（例如在边界区域），可以使用最近邻插值填补
    nan_mask = np.isnan(interpolated_values)
    if np.any(nan_mask):
        interpolated_values[nan_mask] = interpolate.griddata(points, values, missing_points[nan_mask], method='nearest')

    # 将插值结果填回 range_image
    range_image[zero_mask] = interpolated_values

    return range_image

def pc2range(points_array, image_rows, image_cols, ang_start_y, ang_res_y, 
             ang_res_x, max_range, min_range,interpolation=False):
    range_image = np.zeros((image_rows, image_cols, 1), dtype=np.float32)
    x = points_array[:,0]
    y = points_array[:,1]
    z = points_array[:,2]
    # find row id

    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int64(np.round(relative_vertical_angle / ang_res_y))
    # Inverse sign of y for kitti data
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi

    colId = -np.int64((horitontal_angle-90.0)/ang_res_x) + image_cols/2

    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    colId = colId.astype(np.int64)
    # filter range
    thisRange = np.sqrt(x * x + y * y + z * z)
    
    # 这个实测不过滤任何点
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0

    # (rowId < image_rows) 会过滤一些点
    valid_scan = (rowId >= 0) & (rowId < image_rows) & (colId >= 0) & (colId < image_cols)

    rowId_valid = rowId[valid_scan]
    colId_valid = colId[valid_scan]
    thisRange_valid = thisRange[valid_scan]
    # import ipdb
    # ipdb.set_trace()
    range_image[rowId_valid, colId_valid, :] = thisRange_valid.reshape(-1, 1)

    range_image=range_image.squeeze()

    if interpolation:
        range_image=interpolate_range_image(range_image)

    return range_image

def fill_strip(range_image,deviation):
    return range_image,deviation

def pc2range_2(points_array, image_rows, image_cols, ang_start_y, ang_res_y, 
               ang_res_x, max_range, min_range,interpolation=False):
    range_image = np.zeros((image_rows, image_cols, 1), dtype=np.float32)

    x = points_array[:,0]
    y = points_array[:,1]
    z = points_array[:,2]
    # find row id

    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int64(np.round(relative_vertical_angle / ang_res_y))
    # Inverse sign of y for kitti data
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi

    colId = -np.int64((horitontal_angle-90.0)/ang_res_x) + image_cols/2

    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    colId = colId.astype(np.int64)
    # filter range
    thisRange = np.sqrt(x * x + y * y + z * z)
    
    # 这个实测不过滤任何点
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0

    # (rowId < image_rows) 会过滤一些点
    valid_scan = (rowId >= 0) & (rowId < image_rows) & (colId >= 0) & (colId < image_cols)

    rowId_valid = rowId[valid_scan]
    colId_valid = colId[valid_scan]
    thisRange_valid = thisRange[valid_scan]
    # import ipdb
    # ipdb.set_trace()
    range_image[rowId_valid, colId_valid, :] = thisRange_valid.reshape(-1, 1)

    range_image=range_image.squeeze()


    if interpolation:
        range_image=interpolate_range_image(range_image)


    deviation = np.zeros((image_rows,image_cols,3),dtype=np.float32)
    x=x[valid_scan]
    y=y[valid_scan]
    z=z[valid_scan]
    for i in range(len(rowId_valid)):
        fake_vertical_angle=np.float32(rowId_valid[i] * ang_res_y) - ang_start_y
        fake_horizon_angle = - np.float32(colId_valid[i] + 1 - (image_cols/2)) * ang_res_x + 90.0

        fake_horizon_angle=fake_horizon_angle/180.0*np.pi
        fake_vertical_angle=fake_vertical_angle/ 180.0 * np.pi
        fake_z = np.sin(fake_vertical_angle) * thisRange_valid[i]

        fake_x=np.sin(fake_horizon_angle) * np.cos(fake_vertical_angle) * thisRange_valid[i]
        fake_y=np.cos(fake_horizon_angle) * np.cos(fake_vertical_angle) * thisRange_valid[i]
        deviation[rowId_valid[i],colId_valid[i],0]=x[i]-fake_x
        deviation[rowId_valid[i],colId_valid[i],1]=y[i]-fake_y
        deviation[rowId_valid[i],colId_valid[i],2]=z[i]-fake_z

    # range_image,deviation=fill_strip(range_image,deviation)
    return range_image,deviation

def pc2range_kitti(points_array,image_rows=64,interpolation=False):
    image_cols = 1024
    ang_start_y = 24.8
    ang_res_y = 26.8 / (image_rows -1)
    ang_res_x = 360 / image_cols
    max_range = 120
    min_range = 0
    range_img = pc2range_2(points_array, image_rows = image_rows, image_cols = image_cols, 
                           ang_start_y = ang_start_y, ang_res_y = ang_res_y, 
                           ang_res_x = ang_res_x, max_range = max_range, min_range = min_range,
                           interpolation=interpolation)
    return range_img

def range2pc(range_img,deviation, image_rows, image_cols, ang_start_y, ang_res_y, ang_res_x, max_range, min_range):
    rowList = []
    colList = []
    for i in range(image_rows):
        rowList = np.append(rowList, np.ones(image_cols)*i)
        colList = np.append(colList, np.arange(image_cols))

    verticalAngle = np.float32(rowList * ang_res_y) - ang_start_y
    horizonAngle = - np.float32(colList + 1 - (image_cols/2)) * ang_res_x + 90.0
    
    verticalAngle = verticalAngle / 180.0 * np.pi
    horizonAngle = horizonAngle / 180.0 * np.pi


    lengthList = range_img.reshape(image_rows*image_cols)


    if deviation is None:
        x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList
        y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList
        z = np.sin(verticalAngle) * lengthList
    else:
        deviation=deviation.reshape(image_rows*image_cols,3)
        x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList+deviation[...,0]
        y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList+deviation[...,1]
        z = np.sin(verticalAngle) * lengthList+deviation[...,2]
    points = np.column_stack((x,y,z))

    # 删去(0,0,0)
    points=points[~np.all(points == 0, axis=1)]
    return points

def rang2pc_kitti(range_img,deviation,image_rows=64):
    image_cols = 1024
    ang_start_y = 24.8
    ang_res_y = 26.8 / (image_rows -1)
    ang_res_x = 360 / image_cols
    max_range = 120
    min_range = 0
    pc = range2pc(range_img,deviation=deviation, image_rows = image_rows, image_cols = image_cols, ang_start_y = ang_start_y, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)
    return pc

if __name__=="__main__":
    pass
