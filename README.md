# 3DIAS_Pytorch
[project page](https://myavartanoo.github.io/3dias/) [arXiv](https://arxiv.org/abs/2108.08653)

![Example 1](source/airplane.gif)
![Example 2](source/chair1.gif)
![Example 3](source/lamp.gif)
![Example 4](source/speaker.gif)

This repository contains the official code to reproduce the results from the paper: 
**3DIAS: 3D Shape Reconstruction with Implicit Algebraic Surfaces (ICCV 2021)**


## Installation
Clone this repository into any place you want.
```
git clone https://github.com/myavartanoo/3DIAS_PyTorch.git
cd 3DIAS_Pytorch
```

### Dependencies
* Python 3.8.5
* PyTorch >= 1.0.0
* numpy
* Pillow
* open3d
* PyMCubes

Install dependencies in a conda environment.
```
conda create -n 3dias python=3.8
conda activate 3dias

pip install -r requirements.txt
```

### Pretrained model
Download `config.json` and `checkpoint-epoch#.pth` from this [link](https://www.dropbox.com/sh/z7ccstte6i69jju/AABaaCJ9LgKw-JT1Mdf0Tz-ta?dl=0) and save in `weigths` folder.

Note) The provided weight is for the multi-class. To reproduce detailed figures as the animation in this README, you should use the class-wise weights. (It will be distributed soon)


## Quickstart (Demo)
![Example Input](source/example_input.png)

You can now test our code on the provided input images in the `input` folder. (Or you can use other images in shapeNet.)
To this end, simply run, 
```
CUDA_VISIBLE_DEVICES=0 python demo.py --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --inputimg "./input/<image_name>.png"
```
the result meshes are saved in `output` folder. To see the mesh, you can use [meshlab](https://www.meshlab.net/)


If you want to visualize meshes with open3d, run with `--visualize` option as below.
```
CUDA_VISIBLE_DEVICES=0 python demo.py --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --inputimg "./input/<image_name>.png" --visualize
```

<!---
## Dataset
TBD

## Training
TBD

## Testing
TBD
-->

# Citation
If you find our code or paper useful, please consider citing

    @inproceedings{Occupancy Networks,
        title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
        author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
        booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        year = {2019}
    }