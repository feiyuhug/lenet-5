#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
import struct
from numpy import *
from covnet import *
import time


train_covnet = CovNet()

trainim_filepath = '../mnist/train-images-idx3-ubyte'
trainlabel_filepath = '../mnist/train-labels-idx1-ubyte'
trainimfile = open(trainim_filepath, 'rb')
trainlabelfile = open(trainlabel_filepath, 'rb')
train_im = trainimfile.read()
train_label = trainlabelfile.read()
im_index = 0
label_index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , train_im , im_index)
magic, numLabels = struct.unpack_from('>II', train_label, label_index)
print 'train_set:', numImages
im_index += struct.calcsize('>IIII')
label_index += struct.calcsize('>II')

case_num = numImages
for case in range(case_num) :
    im = struct.unpack_from('>784B', train_im, im_index)
    label = struct.unpack_from('>1B', train_label, label_index)
    im_index += struct.calcsize('>784B')
    label_index += struct.calcsize('>1B')
    im = array(im)
    im = im.reshape(28,28)
    bigim = [[-0.1] * 32] * 32
    for i in range(28) :
        for j in range(28) :
            if im[i][j] > 0 :
                bigim[i+2][j+2] = 1.175
    im = array([bigim])
    label = label[0]
    print case, label
    for i in range(10) :
        train_covnet.fw_prop(im, label)
        train_covnet.bw_prop(im, label)
        print(str(train_covnet.outputlay7.maps[0][0][label]))
    train_covnet.print_neterror('./log/error.file')
    break

