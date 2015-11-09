#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
from layer import *

class CovLayer(Layer) :
    def __init__(self, lay_size = [], cov_core_sizes = []) :
        Layer.__init__(self, lay_size)
        self.covcores = []
        for cov_core_size in cov_core_sizes :
            self.covcores.append(ones(cov_core_size)) #网络参数的初始化remain
        self.covcores = array(self.covcores)

    def cov_op(self, pre_maps, covcore_index) :
        pre_map_shape = pre_maps.shape
        covcore_shape = self.covcores[covcore_index].shape
        map_shape = self.maps[covcore_index].shape
        if not (map_shape[-2] == pre_map_shape[-2] - covcore_shape[-2] + 1 \
                and map_shape[-1] == pre_map_shape[-1] - covcore_shape[-1] + 1) :
            return None
        for i in range(map_shape[-2]) :
            for j in range(map_shape[-1]) :
                localrecept = pre_maps[ : , i : i + covcore_shape[-2], j : j + covcore_shape[-1]]
                self.maps[covcore_index][i][j] = sum(localrecept * self.covcores[covcore_index])

    def compute_maps(self, pre_mapset, mapcombflag = False, mapcombindex = []) :
        if not mapcombflag :
            for i in range(len(self.maps)) :
                self.cov_op(pre_mapset, i)
        else :
            for i in range(len(self.maps)) :
                self.cov_op(pre_mapset[mapcombindex[i]], i)





