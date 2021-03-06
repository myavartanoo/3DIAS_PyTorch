import json
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def load_sample_images(fpath):
    images = []

    image_H = Image.open(fpath)
    image_H = np.array(image_H).astype(np.float32)[5:-4,5:-4,:3]

    if image_H is not None:
        images.append(np.transpose(image_H, (2, 0, 1)))

    return np.array(images), fpath.split('/')[-1][:-4]

def gen_polynomial_orders(degree): #4
    under_inds = int((degree)*(degree+1)*(degree+2)/6) # 20 for d=4
    num_inds = int((degree+1)*(degree+2)*(degree+3)/6) # 35 for d=4
    orders = np.zeros([num_inds, 3], dtype=np.float32)
    
    count = 0
    countLargestOrder = under_inds
    for i in range(degree+1):
        for j in range(degree-i+1):
            for k in range(degree-i-j+1):
                if i+j+k==degree:
                    orders[countLargestOrder,:] = np.array([i,j,k], dtype=np.float32)
                    countLargestOrder+=1
                else:
                    orders[count,:] = np.array([i,j,k], dtype=np.float32)
                    count+=1
    return orders # (35,3) for d=4


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
            # self.writer.close()
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
