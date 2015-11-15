#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
from fclayer import *

class OutputLayer(FcLayer) :
    def __init__(self, lay_len, pre_nodesnum) :
        FcLayer.__init__(self, lay_len, pre_nodesnum)
        self.weight = random.choice([-1,1], [lay_len, pre_nodesnum])

    def rbf(self, pre_maps, node_index = -1) :
        pre_nodes = pre_maps.reshape([pre_maps.shape[0] * pre_maps.shape[1] * pre_maps.shape[2]])
        if node_index != -1 :
            self.maps[0][0][node_index] = 0.5 * sum((pre_nodes - self.weight[node_index])**2)
        else :
            for i in range(len(self.maps[0][0])) :
                self.maps[0][0][i] = 0.5 * sum((pre_nodes - self.weight[i])**2)

    def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
        current_error_matrix = array(matrix(list(current_error[0]) * self.weight.shape[1]).T) #current_error代表偏差相对当前层输出的偏导数
        if isweight_update :
            weight_update = (self.weight - array(list(pre_mapset[0]) * self.weight.shape[0])) * current_error_matrix
            self.weight -= learn_rate * weight_update
        pre_error = ((array(list(pre_mapset[0]) * self.weight.shape[0]) - self.weight) * current_error_matrix).sum(axis = 0)
        return pre_error.reshape(pre_mapset.shape) 





