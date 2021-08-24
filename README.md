# 3DIAS_Pytorch
This repository contains the official code to reproduce the results from the paper: 

**3DIAS: 3D Shape Reconstruction with Implicit Algebraic Surfaces (ICCV 2021)**

\[[project page](https://myavartanoo.github.io/3dias/)\] \[[arXiv](https://arxiv.org/abs/2108.08653)\] 

<p align="center">
<img src="source/airplane.gif" width="45%" height="45%"/> <img src="source/lamp.gif" width="45%" height="45%"/> 
<img src="source/speaker.gif" width="45%" height="45%"/> <img src="source/chair.gif" width="45%" height="45%"/> 
</p>


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
* PyMCubes (or build this [repo](https://github.com/tatsy/torchmcubes))

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
<p align="center">
  <img src="source/chair_rotate.gif" width="30%" height="30%">
</p>

You can now test our demo code on the provided input images in the `input` folder. (Or you can use other images in shapeNet.)
To this end, simply run, 
```
CUDA_VISIBLE_DEVICES=0 python demo.py --inputimg "./input/<image_name>.png" --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" 
```
The result meshes are saved in `output` folder. (We've created a few example meshes)
* total.ply is a whole mesh
* parts_<number>.ply are meshes for parts
To see the mesh, you can use [meshlab](https://www.meshlab.net/)

If you want to visualize meshes with open3d, run with `--visualize` option as below.
```
CUDA_VISIBLE_DEVICES=0 python demo.py --inputimg "./input/<image_name>.png" --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --visualize
```

The preprocessed dataset, training, testing code will be distributed soon.
<!---
## Dataset
TBD

## Training
TBD

## Testing
TBD
-->

## Citation
If you find our code or paper useful, please consider citing

    @inproceedings{Occupancy Networks,
        title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
        author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
        booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        year = {2019}
    }
