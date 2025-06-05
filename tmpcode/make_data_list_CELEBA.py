data_path="/amax/lm/Datahouse/CELEBA/data"

data_list_path="/amax/lm/Datahouse/CELEBA/train_data.txt"

import os
dataname=os.listdir(data_path)


with open(data_list_path,"w",encoding="utf-8") as f:
    for name in dataname:
        f.write(os.path.join(data_path,name)+"\n")
import ipdb
ipdb.set_trace()