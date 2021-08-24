### Demo: generate mesh
# CUDA_VISIBLE_DEVICES=0 python demo.py --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --inputimg "./input/03001627_ef4e47e54bfc685cb40f0ac0fb9a650d_14.png"

### Demo: generate mesh and visualize with open3D
CUDA_VISIBLE_DEVICES=0 python demo.py --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --inputimg "./input/03001627_ef4e47e54bfc685cb40f0ac0fb9a650d_14.png" --visualize