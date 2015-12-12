#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
from layer import *

class CovLayer(Layer) :
    def __init__(self, lay_size = [], cov_core_sizes = [], mapcombindex = []) :
        Layer.__init__(self, lay_size)
        self.covcores = []
        self.covbias = []
        self.mapcombindex = mapcombindex
        for cov_core_size in cov_core_sizes :
            Fi = cov_core_size[0] * cov_core_size[1] + 1
            self.covcores.append(random.uniform(-2.4/Fi, 2.4/Fi, cov_core_size)) #网络参数的初始化
            self.covbias.append(random.uniform(-2.4/Fi, 2.4/Fi)) #卷积操作的bias
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
                val = sum(localrecept * self.covcores[covcore_index]) + self.covbias[covcore_index]
                val = exp((4.0/3)*val) 
                self.maps[covcore_index][i][j] = 1.7159 * (val - 1) / (val + 1) 

    def calc_maps(self, pre_mapset, mapcombflag = False) :
        if not mapcombflag :
            for i in range(len(self.maps)) :
                self.cov_op(pre_mapset, i)
        else :
            for i in range(len(self.maps)) :
                self.cov_op(pre_mapset[self.mapcombindex[i]], i)

    def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
        self.current_error = current_error
        selfmap_line = self.maps.reshape([self.maps.shape[0] * self.maps.shape[1] * self.maps.shape[2]])
        currenterror_line = current_error.reshape([current_error.shape[0] * current_error.shape[1] * current_error.shape[2]])
        pcurrent_error = array([((2.0/3)*(1.7159 - (1/1.7159) * selfmap_line[i]**2))*currenterror_line[i]\
                for i in range(len(selfmap_line))]).reshape(self.maps.shape)
        weight_update = self.covcores * 0
        bias_update = zeros([len(self.covbias)])
        pre_error = zeros(pre_mapset.shape)
        for i in range(self.maps.shape[0]) :
            if self.mapcombindex != [] :
                pre_maps = pre_mapset[self.mapcombindex[i]]
                select_pre_error = pre_error[self.mapcombindex[i]]
            else :
                pre_maps = pre_mapset
                select_pre_error = pre_error 
            for mi in range(self.maps.shape[1]) :
                for mj in range(self.maps.shape[2]) :
                    cov_maps = pre_maps[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]
                    weight_update[i] += cov_maps * pcurrent_error[i][mi][mj]
                    bias_update[i] += pcurrent_error[i][mi][mj]
                    select_pre_error[:, mi:mi+self.covcores[i].shape[1], mj:mj+self.covcores[i].shape[2]]\
                            += self.covcores[i] * pcurrent_error[i][mi][mj]
        if isweight_update :
            self.covcores -= learn_rate * weight_update
            self.covbias -= learn_rate * bias_update
        return pre_error



