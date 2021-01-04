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

class SegMasksTestLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(max(1,int(np.sum(bottom[1].data[...]))),1,160,160)
        top[1].reshape(max(1,int(np.sum(bottom[2].data[...]))),1,160,160)
        top[2].reshape(max(1,int(np.sum(bottom[3].data[...]))),1,160,160)
        top[3].reshape(max(1,int(np.sum(bottom[4].data[...]))),1,160,160)
        top[4].reshape(max(1,int(np.sum(bottom[5].data[...]))),1,160,160)
        top[5].reshape(max(1,int(np.sum(bottom[6].data[...]))),1,160,160)
        top[6].reshape(max(1,int(np.sum(bottom[7].data[...]))),1,160,160)
        top[7].reshape(max(1,int(np.sum(bottom[8].data[...]))),1,160,160)

    def forward(self, bottom, top):
        allMasks = bottom[0].data[...]
        len8 = int(np.sum(bottom[1].data[...]))
        len16 = int(np.sum(bottom[2].data[...]))
        len24 = int(np.sum(bottom[3].data[...]))
        len32 = int(np.sum(bottom[4].data[...]))
        len48 = int(np.sum(bottom[5].data[...]))
        len64 = int(np.sum(bottom[6].data[...]))
        len96 = int(np.sum(bottom[7].data[...]))
        len128 = int(np.sum(bottom[8].data[...]))
        
        top[0].reshape(max(1,len8),1,160,160)
        top[1].reshape(max(1,len16),1,160,160)
        top[2].reshape(max(1,len24),1,160,160)
        top[3].reshape(max(1,len32),1,160,160)
        top[4].reshape(max(1,len48),1,160,160)
        top[5].reshape(max(1,len64),1,160,160)
        top[6].reshape(max(1,len96),1,160,160)
        top[7].reshape(max(1,len128),1,160,160)
        
        
        firstIndex = 0
        for i,lenScale in enumerate([len8, len16, len24, len32, len48, len64, len96, len128]):
            if lenScale>0:
                top[i].data[...]=allMasks[firstIndex:firstIndex+lenScale]
                firstIndex+=lenScale
            else:
                top[i].data[...]=-1
        

    def backward(self, top, propagate_down, bottom):
        pass
    
