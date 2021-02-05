import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time


with open('trt_pose_data/human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


MODEL_WEIGHTS = 'trt_pose_data/resnet18_baseline_att_224x224_A_epoch_249.pth'
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS))

OPTIMIZED_MODEL = 'trt_pose_data/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

cmap, paf = model(data)
cmap_trt, paf_trt = model_trt(data)
# check the output against PyTorch
print(torch.max(torch.abs(paf - paf_trt)))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))
