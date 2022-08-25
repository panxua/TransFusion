import torch
# models/resnet50.pth "work_dirs/bevtransfusion_vod_voxel_C/300obj_fixed/epoch_18.pth" #"work_dirs/transfusion_vod_voxel_LC/trained/epoch_14.pth"
img_file = "models/resnet50.pth"  #"work_dirs/transfusion_vod_C/LSSFPN_e23/epoch_23.pth"
pts_file = "models/transfusion_l_epoch_5.pth"
output_file = "models/baseline_L.pth"

img = torch.load(img_file, map_location='cpu')
pts = torch.load(pts_file, map_location='cpu')
new_model = {"state_dict": pts["state_dict"]}

if 'state_dict' not in img:
    for k,v in img.items():
        new_model['state_dict']["img_backbone."+k] = v
        new_model['state_dict']['img_bev_encoder_backbone.'+k] = v
else:
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
