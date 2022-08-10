import torch
img_file = "work_dirs/bevtransfusion_vod_voxel_C/300obj_fixed/epoch_18.pth" #"work_dirs/transfusion_vod_voxel_LC/trained/epoch_14.pth"
# img_file = "models/nuScenes_3Ddetection_e140.pth"
pts_file = "models/transfusionL_fade_e18.pth"
output_file = "models/bevfusion_fade_e18_retrained.pth"

img = torch.load(img_file, map_location='cpu')
pts = torch.load(pts_file, map_location='cpu')
new_model = {"state_dict": pts["state_dict"]}


for k,v in img["state_dict"].items():
    if 'pts' in k:
        continue
    if 'img_' in k:
        new_model["state_dict"][k] = v
    elif 'backbone' in k or 'neck' in k:
        new_model["state_dict"]['img_'+k] = v
        new_model["state_dict"]['img_bev_'+k] = v
        print('img_'+k)
        print('img_bev_'+k)
torch.save(new_model, output_file)
