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

class SplitSegFlags(caffe.Layer):
    
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[1].data[...].shape)
        top[1].reshape(*bottom[2].data[...].shape)
        top[2].reshape(*bottom[3].data[...].shape)
        top[3].reshape(*bottom[4].data[...].shape)
        top[4].reshape(*bottom[5].data[...].shape)
        top[5].reshape(*bottom[6].data[...].shape)
        top[6].reshape(*bottom[7].data[...].shape)
        top[7].reshape(*bottom[8].data[...].shape)
        
        top[0+8].reshape(*bottom[1].data[...].shape)
        top[1+8].reshape(*bottom[2].data[...].shape)
        top[2+8].reshape(*bottom[3].data[...].shape)
        top[3+8].reshape(*bottom[4].data[...].shape)
        top[4+8].reshape(*bottom[5].data[...].shape)
        top[5+8].reshape(*bottom[6].data[...].shape)
        top[6+8].reshape(*bottom[7].data[...].shape)
        top[7+8].reshape(*bottom[8].data[...].shape)

    def forward(self, bottom, top):
        allFlags = bottom[0].data[...]
        
        top[0].reshape(*bottom[1].data[...].shape)
        top[1].reshape(*bottom[2].data[...].shape)
        top[2].reshape(*bottom[3].data[...].shape)
        top[3].reshape(*bottom[4].data[...].shape)
        top[4].reshape(*bottom[5].data[...].shape)
        top[5].reshape(*bottom[6].data[...].shape)
        top[6].reshape(*bottom[7].data[...].shape)
        top[7].reshape(*bottom[8].data[...].shape)
        
        top[0+8].reshape(*bottom[1].data[...].shape)
        top[1+8].reshape(*bottom[2].data[...].shape)
        top[2+8].reshape(*bottom[3].data[...].shape)
        top[3+8].reshape(*bottom[4].data[...].shape)
        top[4+8].reshape(*bottom[5].data[...].shape)
        top[5+8].reshape(*bottom[6].data[...].shape)
        top[6+8].reshape(*bottom[7].data[...].shape)
        top[7+8].reshape(*bottom[8].data[...].shape)
        
        numFlagsPerScale = bottom[1].data[...].shape[3]
        
        top[0].data[...] = allFlags[:,:,:,:numFlagsPerScale]
        top[1].data[...] = allFlags[:,:,:,numFlagsPerScale:numFlagsPerScale*2]
        top[2].data[...] = allFlags[:,:,:,numFlagsPerScale*2:numFlagsPerScale*3]
        top[3].data[...] = allFlags[:,:,:,numFlagsPerScale*3:numFlagsPerScale*4]
        top[4].data[...] = allFlags[:,:,:,numFlagsPerScale*4:numFlagsPerScale*5]
        top[5].data[...] = allFlags[:,:,:,numFlagsPerScale*5:numFlagsPerScale*6]
        top[6].data[...] = allFlags[:,:,:,numFlagsPerScale*6:numFlagsPerScale*7]
        top[7].data[...] = allFlags[:,:,:,numFlagsPerScale*7:]
        
        if np.sum(allFlags[:,:,:,:numFlagsPerScale]) == 0:
            top[0+8].data[...] = allFlags[:,:,:,:numFlagsPerScale]    
            top[0+8].data[0,0,0,0]=1
        else:
            top[0+8].data[...] = allFlags[:,:,:,:numFlagsPerScale]    
            
        if np.sum(allFlags[:,:,:,numFlagsPerScale:numFlagsPerScale*2]) == 0:
            top[1+8].data[...] = allFlags[:,:,:,numFlagsPerScale:numFlagsPerScale*2]  
            top[1+8].data[0,0,0,0]=1
        else:
            top[1+8].data[...] = allFlags[:,:,:,numFlagsPerScale:numFlagsPerScale*2]   
            
        if np.sum(allFlags[:,:,:,numFlagsPerScale*2:numFlagsPerScale*3]) == 0:
            top[2+8].data[...] = allFlags[:,:,:,numFlagsPerScale*2:numFlagsPerScale*3]  
            top[2+8].data[0,0,0,0]=1
        else:
            top[2+8].data[...] = allFlags[:,:,:,numFlagsPerScale*2:numFlagsPerScale*3]
            
        if np.sum(allFlags[:,:,:,numFlagsPerScale*3:numFlagsPerScale*4]) == 0:
            top[3+8].data[...] = allFlags[:,:,:,numFlagsPerScale*3:numFlagsPerScale*4]  
            top[3+8].data[0,0,0,0]=1
        else:
            top[3+8].data[...] = allFlags[:,:,:,numFlagsPerScale*3:numFlagsPerScale*4]
            
        if np.sum(allFlags[:,:,:,numFlagsPerScale*4:numFlagsPerScale*5]) == 0:
            top[4+8].data[...] = allFlags[:,:,:,numFlagsPerScale*4:numFlagsPerScale*5]  
            top[4+8].data[0,0,0,0]=1
        else:
            top[4+8].data[...] = allFlags[:,:,:,numFlagsPerScale*4:numFlagsPerScale*5]
            
        if np.sum(allFlags[:,:,:,numFlagsPerScale*5:numFlagsPerScale*6]) == 0:
            top[5+8].data[...] = allFlags[:,:,:,numFlagsPerScale*5:numFlagsPerScale*6]  
            top[5+8].data[0,0,0,0]=1
        else:
            top[5+8].data[...] = allFlags[:,:,:,numFlagsPerScale*5:numFlagsPerScale*6]
            
        if np.sum(allFlags[:,:,:,numFlagsPerScale*6:numFlagsPerScale*7]) == 0:
            top[6+8].data[...] = allFlags[:,:,:,numFlagsPerScale*6:numFlagsPerScale*7]  
            top[6+8].data[0,0,0,0]=1
        else:
            top[6+8].data[...] = allFlags[:,:,:,numFlagsPerScale*6:numFlagsPerScale*7]
            
        if np.sum(allFlags[:,:,:,numFlagsPerScale*7:]) == 0:
            top[7+8].data[...] = allFlags[:,:,:,numFlagsPerScale*7:]  
            top[7+8].data[0,0,0,0]=1
        else:
            top[7+8].data[...] = allFlags[:,:,:,numFlagsPerScale*7:]
            

    def backward(self, top, propagate_down, bottom):
        pass
    
