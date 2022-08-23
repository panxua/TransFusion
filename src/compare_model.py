import torch
# models/resnet50.pth "work_dirs/bevtransfusion_vod_voxel_C/300obj_fixed/epoch_18.pth" #"work_dirs/transfusion_vod_voxel_LC/trained/epoch_14.pth"
file1 = "models/bevfusion_formal_trained_C.pth"  #'models/fusion_model_retrained.pth' 
file2 =  "work_dirs/bevtransfusion_vod_voxel_LC_BP/epoch_12.pth" #"work_dirs/bevtransfusion_vod_voxel_LC/epoch_1.pth" #"work_dirs/transfusion_vod_voxel_LC/trained/epoch_17.pth" 

model1 = torch.load(file1, map_location='cpu')
model2 = torch.load(file2, map_location='cpu')

model1_keys = set(model1['state_dict'].keys())
model2_keys = set(model2['state_dict'].keys())

print("\nmodel1:\n", model1_keys-model2_keys)
print("model2:\n", model2_keys-model1_keys)
print("common:\n", model1_keys&model2_keys)

print()
for name, p in model1['state_dict'].items():
    if name in model2['state_dict']:
        if torch.any(p != model2['state_dict'][name]):
            print(name)


