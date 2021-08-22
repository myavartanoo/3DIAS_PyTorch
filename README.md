# 3DIAS_Pytorch
[project page](https://myavartanoo.github.io/3dias/) [arXiv](https://arxiv.org/abs/2108.08653) [ICCV]()

![Example 1](source/airplane.gif)
![Example 2](source/chair1.gif)
![Example 3](source/lamp.gif)
![Example 4](source/speaker.gif)

This repository contains the official code to reproduce the results from the paper: 
**3DIAS: 3D Shape Reconstruction with Implicit Algebraic Surfaces (ICCV 2021)**


## Installation
TBD

### Pretrained model
save `config.json` and `checkpoint.pth` in `weigths` folder


## Demo
![Example Input](source/example_input.png)

You can now test our code on the provided input images in the `input` folder.
To this end, simply run `sh run.sh` or, 
```
CUDA_VISIBLE_DEVICES=0 python demo.py --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --inputimg "./input/03001627_ef4e47e54bfc685cb40f0ac0fb9a650d_14.png"
```


## Dataset
TBD

## Training
TBD

## Testing
TBD


# Citation
If you find our code or paper useful, please consider citing

    @inproceedings{Occupancy Networks,
        title = {Occupancy Networks: Learning 3D Reconstruction in Function Space},
        author = {Mescheder, Lars and Oechsle, Michael and Niemeyer, Michael and Nowozin, Sebastian and Geiger, Andreas},
        booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        year = {2019}
    }