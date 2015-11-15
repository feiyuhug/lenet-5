#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
from layer import *

class FcLayer(Layer) :
    def __init__(self, lay_len, pre_nodesnum) :
        Layer.__init__(self, [[1, lay_len]])
        Fi = pre_nodesnum + 1
        self.weight = random.uniform(-2.4/Fi, 2.4/Fi, [lay_len, pre_nodesnum])
        self.bias = random.uniform(-2.4/Fi, 2.4/Fi, [lay_len])
    
    def fc_op(self, pre_maps, node_index) :
        pre_nodes = pre_maps.reshape([pre_maps.shape[0] * pre_maps.shape[1] * pre_maps.shape[2]])
        val  = sum(self.weight[node_index] * pre_nodes) + self.bias[node_index]
        val = exp((4.0/3)*val)
        self.maps[0][0][node_index] = 1.7159 *  (val -1) / (val + 1)

    def calc_maps(self, pre_mapset) :
        for i in range(len(self.maps[0][0])) :
            self.fc_op(pre_mapset, i)

    def back_propa(self, pre_mapset, current_error, learn_rate, isweight_update) :
	pcurrent_error = [((2.0/3)*(1.7159 - (1/1.7159) * self.maps[0][0][i]**2))*current_error[0][0][i]\
                for i in range(self.maps.shape[-1])]
	weight_update = dot(matrix(pcurrent_error).T, \
			matrix(pre_mapset.reshape([1, pre_mapset.shape[0] * pre_mapset.shape[1] * pre_mapset.shape[2]])))
	bias_update = array(pcurrent_error)
        if isweight_update :
	    self.weight -= learn_rate * weight_update
	    self.bias -= learn_rate * bias_update
	pre_error = array(dot(matrix(pcurrent_error), matrix(self.weight))).reshape(pre_mapset.shape)
        return pre_error




