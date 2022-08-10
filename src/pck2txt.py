import mmcv
from tqdm import tqdm
import os

file_paths = ["output/bev_C_pts_bbox.pkl"]
result_paths = ["output/TL_val/"]
for i, path in enumerate(file_paths):
    results = mmcv.load(path)
    if not os.path.exists(result_paths[i]):
        os.mkdir(result_paths[i])
    for ret in tqdm(results):
        if not len(ret['name']):
            continue
        sample_idx = ret['sample_idx'][0]
        file = open(result_paths[i]+"%05d.txt"%sample_idx,"w")
        for obj_idx, obj_name in tqdm(enumerate(ret['name']), total = len(ret['name']), leave=False):
            temp_ret = [obj_name, str(ret['truncated'][obj_idx]), str(ret['occluded'][obj_idx]), \
                str(ret['alpha'][obj_idx]), *map(str, ret['bbox'][obj_idx]), *map(str, ret['dimensions'][obj_idx]), \
                *map(str, ret['location'][obj_idx]), str(ret['rotation_y'][obj_idx]), str(ret['score'][obj_idx]),]
            file.write(" ".join(temp_ret)+"\n")

        
