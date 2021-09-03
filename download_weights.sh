#!/bin/bash

for f in "02691156_airplane" "02828884_bench" "02933112_cabinet" "02958343_car" "03001627_chair" "03211117_display" "03636649_lamp" "03691459_speaker" "04090263_rifle" "04256520_sofa" "04379243_table" "04401088_phone" "04530566_vessel"
do
wget "http://data.cv.snu.ac.kr:8008/webdav/dataset/3DIAS/single_class/${f}.zip"
done