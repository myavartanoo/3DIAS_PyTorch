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
* PyTorch 1.7.1
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
Download `config.json` and `checkpoint-epoch#.pth` from below links and save in `weigths` folder.
Note that we get `Multi-class` weight by training with all-classes and `Single-class` weight by training with each class

#### Multi-class
> [Dropbox](https://www.dropbox.com/sh/z7ccstte6i69jju/AABaaCJ9LgKw-JT1Mdf0Tz-ta?dl=0) or [Mirror](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/multi_class.zip)

#### Single-class
To download all the single-class weigths, run
```
sh download_weights.sh
```

Or you can get the weights one-by-one.
> [airplane](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/02691156_airplane.zip) / [bench](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/02828884_bench.zip) / [cabinet](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/02933112_cabinet.zip) / [car](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/02958343_car.zip) / [chair](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/03001627_chair.zip) / [display](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/03211117_display.zip) / [lamp](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/03636649_lamp.zip) / [speaker](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/03691459_speaker.zip) / [rifle](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/04090263_rifle.zip) / [sofa](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/04256520_sofa.zip) / [table](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/04379243_table.zip) / [phone](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/04401088_phone.zip) / [vessel](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/04530566_vessel.zip)




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
python demo.py --device "0" --inputimg "./input/<image_name>.png" --config "./weights/config.json" --resume "./weights/checkpoint-epoch890.pth" --visualize
```

The preprocessed dataset, training, testing code will be distributed soon.

## Dataset
Dowload below two zip files and unzip in `data` folder.
[images](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/images.zip)
[newDataPoints](http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/newDataPoints.zip)

## Training
To run the training code, 
```
python train.py --device "0" --config config.json --tag "exp_name"
```
Note that,
1. the log and model will be saved at `trainer/save_dir` in `config.json`
2. `--tag` is for the name of experiment

## Testing
For each experiment, there are `config.json` file and `checkpoint-epoch###.pth` file.
```
python test.py --device "0" --config /path/to/saved_config/config.json --resume "/path/to/saved_model/checkpoint-epoch###.pth"
```

## Citation
If you find our code or paper useful, please consider citing

    @inproceedings{3DIAS,
        title = {3DIAS: 3D Shape Reconstruction with Implicit Algebraic Surfaces},
        author = {Mohsen Yavartanoo, JaeYoung Chung, Reyhaneh Neshatavar, Kyoung Mu Lee},
        booktitle = {Proceedings IEEE Conf. on International Conference on Computer Vision (ICCV)},
        year = {2021}
    }
