import os
path="/amax/lm/Datahouse/KITTI/val_16_64"
with open(os.path.join(path,"meta_data.txt"),"w",encoding="utf-8") as f:
    name_lst=os.listdir(os.path.join(path,"hr_64"))
    for it in name_lst:
        f.write(it+","+it+","+it+"\n")

print("OK")
# import ipdb
# ipdb.set_trace()