'''
@author Christian Wilms
@date 01/05/21
'''

import sys
import os
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.getcwd())
import caffe
                        
class ReshapeLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.numParis = min(bottom[0].shape[0],bottom[1].shape[0])
        top[0].reshape(self.numParis, 1,1,1)

    def forward(self, bottom, top):
        top[0].reshape(*bottom[0].shape)
        top[0].data[...]=bottom[0].data
        
        
    def backward(self, top, propagate_down, bottom):
        pass
