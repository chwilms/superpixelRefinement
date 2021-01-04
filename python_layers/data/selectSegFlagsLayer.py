'''
@author Christian Wilms
@date 01/05/21
'''

import sys
import os
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.getcwd())
import caffe

import numpy as np

class SelectSegFlagsLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data[...].shape)

    def forward(self, bottom, top):
        flags = bottom[0].data[0,0,0,:]
        topK = bottom[1].data[:,0,0,0].astype(np.int)
        
        onesInFlagsIds = np.nonzero(flags)[0]
        newOneInFlags = np.zeros(len(onesInFlagsIds))
        newOneInFlags[topK]=1
        flags[onesInFlagsIds]=newOneInFlags
        
        top[0].reshape(*bottom[0].data[...].shape)
        top[0].data[0,0,0,:]=flags

    def backward(self, top, propagate_down, bottom):
        pass
    
