from __future__ import print_function, division
from time import time
import numpy as np
import pandas as pd
from PIL import Image
import torch

from torch.utils.data import Dataset
from base import BaseDataLoader
from PIL import Image


class Shape3DLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, metadata, point_limit, batch_size, task, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.metadata = metadata
        self.data_dir = data_dir
        self.point_limit = point_limit

        self.dataset = Shape3DDataset(self.metadata, self.data_dir, self.point_limit, task=task, transform=None)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Shape3DDataset(Dataset):
    """Dataset for 3D shape generation"""

    def __init__(self, csv_file, data_dir, point_limit, task, transform=None):
        """ 
        csv_file: location of metadata.csv file
        data_dir: path where the data located
        point_limit: list of limit #points - on, out, in (in order)
        transform: Ignore this
        """
        self.task = task

        self.data_dir = data_dir
        metadata = pd.read_csv(csv_file)
        if self.task == "train":
            self.cls_sizes = np.array(metadata['class_size'])
        elif self.task == "val":
            self.cls_sizes = np.array(metadata['class_size_valid'])
        elif self.task == "test":
            self.cls_sizes = np.array(metadata['class_size_test'])
        self.cls_names = np.array(metadata['class_name'])
        self.cls_num = np.array(metadata['class_num'])
        

        self.point_limit = point_limit # on, out, in 
        self.indices = self._indices_generator()
        self.allpoints = np.load(data_dir + 'newDataPoints/all.npy') # (1,000,000 , 3)
        self.transform = transform
        

        self.loclist = ['on','out','in']
        self.planelist = ['xy','yz','xz']
        
    def __len__(self):
        return np.sum(self.cls_sizes)

    def __getitem__(self, index):
        def _adjust_points(points, limit_num):
            """ if limit_num is larger than the #points in .ply, use replace sampling """
            idx = np.random.choice(np.arange(points.shape[0]), size=limit_num, replace=(points.shape[0]<limit_num))
            return points[idx,:]

        def _inout_points_loader(class_name, number_in_class):
            """ Instead of saving in/out points, We save all the location of in/out points to all.npy and also save the index for inside points at `id` folder """
            points = {}
            in_points_path = self.data_dir + 'newDataPoints/id/' + class_name + '/'+self.task+'/' + str(number_in_class).zfill(4) + '.npy'
            in_idx = np.load(in_points_path)

            spRate = 10000.0/self.allpoints.shape[0]

            num_inpoints  = int(np.round(in_idx.shape[0] * spRate))
            num_outpoints = 10000 - num_inpoints
            assert num_inpoints > 0
            assert num_outpoints > 0

            points['in'] = _adjust_points(self.allpoints[in_idx,:], num_inpoints)
            points['out'] = _adjust_points(np.delete(self.allpoints, in_idx, axis=0), num_outpoints)

            return points, num_inpoints

        def _on_points_loader(class_name, number_in_class):
            points = {}
            in_points_path = self.data_dir + 'newDataPoints/on_10/' + class_name + '/'+self.task+'/' + str(number_in_class).zfill(4) + '.npy'
            on_points_normal = _adjust_points(np.load(in_points_path), self.point_limit[0])
            points['on'] = on_points_normal[:,0:3]
            normal = on_points_normal[:,3:6]
            return points, normal

        def _image_loader(class_name, number_in_class, img_rand_idx):
            image_H = Image.open(self.data_dir + 'images/img/'+class_name+'/'+self.task+'/'+str(number_in_class).zfill(4)+'/'+str(img_rand_idx).zfill(2)+'.png') # (137, 137, 4)
            image_H = np.array(image_H).astype(np.float32)[5:-4,5:-4,:3] 
            return np.transpose(image_H, (2, 0, 1)) # (batch, 3, 128, 128)

        if torch.is_tensor(index):
            index = index.tolist()

        # select class, #data
        number_in_class = self.indices[index,0]
        class_name = str(self.indices[index,1]).zfill(8)
        
        # loading random image
        img_rand_idx = np.random.randint(24, size=1)[0] # 24 random images
        image_H = _image_loader(class_name, number_in_class, img_rand_idx)
        if image_H.size==0: print("class:{}, num:{}".format(class_name, number_in_class)); raise

        # loading points, normal vector
        inout_points, num_inpoints = _inout_points_loader(class_name, number_in_class)
        on_points, normal = _on_points_loader(class_name, number_in_class)
        if on_points['on'].size==0: print("class:{}, num:{}".format(class_name, number_in_class)); raise
        if inout_points['in'].size==0: print("class:{}, num:{}".format(class_name, number_in_class)); raise
        if inout_points['out'].size==0: print("class:{}, num:{}".format(class_name, number_in_class)); raise

        directory = np.array([int(class_name), number_in_class])
        target = {'onpts': on_points['on'], 
                  'normal': normal,
                  'inoutpts': np.concatenate((inout_points['in'], inout_points['out']), axis=0), # in-out
                  'numinside': num_inpoints, # in + out = 10k
                  'class_num': self.indices[index,2].astype(np.int_),
                  'directory': directory # list, not Tensor
                  }

        return image_H, target

    def _indices_generator(self):
        indices = np.zeros([sum(self.cls_sizes), 3])
        c = 0
        for ind in range(len(self.cls_sizes)):
            indices[c:self.cls_sizes[ind] + c,0] = np.arange(self.cls_sizes[ind])
            indices[c:self.cls_sizes[ind] + c,1] = self.cls_names[ind]
            indices[c:self.cls_sizes[ind] + c,2] = self.cls_num[ind]
            c = c + self.cls_sizes[ind]
        return indices.astype(int)