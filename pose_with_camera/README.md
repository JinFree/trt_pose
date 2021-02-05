1. download resnet18_baseline_att_224x224_A in trt_pose_data


| Model | Jetson Nano | Jetson Xavier | Weights |
|-------|-------------|---------------|---------|
| resnet18_baseline_att_224x224_A | 22 | 251 | [download (81MB)](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) |
| densenet121_baseline_att_256x256_B | 12 | 101 | [download (84MB)](https://drive.google.com/open?id=13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU) |

2. execure trt optimization

python3 trt_pose-convert_torch2trt.py

3. validate optimization

python3 trt_pose-convert-test.py

4. inference pose with cam

python3 inference_pose.py

- need to check camera number with 

ls /dev/video*

- if csi camera is installes, need to check camera number

v4l2-ctl -d 0 --all
