# 设置源文件和目标文件路径
input_file = '/amax/lm/Datahouse/KITTI/val_16_64/meta_data.txt'      # 需要读取的原文件
output_file = '/amax/lm/Datahouse/KITTI/val_16_64/meta_data_sub.txt'  # 要写入的新文件

# 打开源文件读取，目标文件写入
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for i, line in enumerate(infile):
        if i >= 10:
            break
        outfile.write(line)