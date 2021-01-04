'''
@author Christian Wilms
@date 01/05/21
'''

import sys
import os
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.getcwd())
import caffe

import json
import numpy as np

class CreateGTLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        params = eval(self.param_str)
        scale = str(int(params["scale"]))
        with open('spxGT_train2014_FH_'+scale+'.json') as f:
            self.data=json.load(f)

    def reshape(self, bottom, top):
        spxMaps = bottom[0].data[...]
        self.numSampledSpx = 0
        self.uniques = []
        for i in range(spxMaps.shape[0]):
            self.uniques.append(np.unique(spxMaps[i,0,:,:]))
            self.numSampledSpx+=len(self.uniques[-1])
        
        top[0].reshape(self.numSampledSpx,1,1,1)

    def forward(self, bottom, top):
        spxMaps = bottom[0].data[...]
        maskIDs = bottom[1].data[...].flatten().astype(np.int)
        
        result = np.zeros((self.numSampledSpx))
        lowerIndex = 0
        for i in range(spxMaps.shape[0]):
            spxIds = self.uniques[i]
            maskID = maskIDs[i]
            strMaskID = str(maskID)
            numSpx = len(spxIds)
            if maskID == -1 or not self.data.has_key(strMaskID):
                result[lowerIndex:lowerIndex+numSpx]=-1
                if maskID != -1:
                    print 'key not found', maskID
            else:
                annSpxIds = self.data[strMaskID]
                result[lowerIndex:lowerIndex+numSpx] = np.isin(spxIds, annSpxIds)
            lowerIndex+=numSpx
            
        top[0].data[...]=result.reshape((-1,1,1,1))
        top[0].reshape(self.numSampledSpx,1,1,1)
        
    def backward(self, top, propagate_down, bottom):
        pass

