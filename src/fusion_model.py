import torch
img_file = "/home/xuanyu/radarfusion/TransFusion/work_dirs/transfusion_vod_C/epoch_35.pth"
# img_file = "models/nuScenes_3Ddetection_e140.pth"
pts_file = "models/transfusionL_fade_e18.pth"
output_file = "models/fusion_model_retrained.pth"

img = torch.load(img_file, map_location='cpu')
pts = torch.load(pts_file, map_location='cpu')
new_model = {"state_dict": pts["state_dict"]}
for k,v in img["state_dict"].items():
    if 'backbone' in k or 'neck' in k:
        new_model["state_dict"]['img_'+k] = v
        print('img_'+k)

torch.save(new_model, output_file)
