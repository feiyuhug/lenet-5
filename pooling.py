#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *

class PoolingLayer(Layer) :
    def __init__(self, lay_size = [], pool_core_sizes = []) :
        Layer.__init__(self, lay_size)
        self.poolparas = ones([len(lay_size), 2])
        self.poolcore_sizes = array(pool_core_sizes)
    
    def pool_op(self, pre_map, pool) :
        pre_map_shape = pre_map.shape
        

    def compute_maps(self, pre_mapset) :
        for i in range(len(self.maps)) :
            
        





