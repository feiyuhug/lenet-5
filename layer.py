#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *

class Layer(object) :
    def __init__(self, lay_size = []) :
        self.lay_size = lay_size
        self.maps = []
        for map_size in lay_size :
            self.maps.append(zeros(map_size))
        self.maps = array(self.maps)
    
        





