### Demo: generate mesh
# CUDA_VISIBLE_DEVICES=0 python demo.py --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --inputimg "./input/<image_name>.png"

### Demo: generate mesh and visualize with open3D
# CUDA_VISIBLE_DEVICES=0 python demo.py --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --inputimg "./input/<image_name>.png" --visualize