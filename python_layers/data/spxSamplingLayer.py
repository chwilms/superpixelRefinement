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

class SpxSamplingLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.numSampledSpx = bottom[1].shape[0]
        self.numFeatures = bottom[0].shape[3]
        
        pooledSegs = bottom[2].data[...].astype(np.int)
        numSegs = 0
        self.uniques = []
        for i in range(pooledSegs.shape[0]):
            self.uniques.append(np.unique(pooledSegs[i]))
            numSegs+=len(self.uniques[-1])
        self.numberOfPairs = max(1,numSegs)
        top[0].reshape(self.numberOfPairs,1,1,self.numFeatures+1)
        top[1].reshape(self.numberOfPairs,1,1,3)

    def forward(self, bottom, top):
        spxFeat = bottom[0].data[...][0,0]
        attWeights = bottom[1].data[...]
        if len(attWeights.shape)==4:
            attWeights=attWeights[:,0,0:]
        
        params = eval(self.param_str)
        scale = params["scale"]
        
        top[0].reshape(self.numberOfPairs,1,1,self.numFeatures+1)
        top[1].reshape(self.numberOfPairs,1,1,3)
        
        topIndex = 0
        
        pooledSegs = bottom[2].data[...].astype(np.int)
        
        for i in range(self.numSampledSpx):
            attVector = attWeights[i,:].reshape((-1))
            indices = self.uniques[i]
            assert np.all(self.uniques[i] == np.unique(pooledSegs[i]))
            numSamples = len(indices)
            top[0].data[topIndex:topIndex+numSamples,0,0,:-1] = spxFeat[indices,:]
            top[0].data[topIndex:topIndex+numSamples,0,0,-1] = attVector[indices]
            top[1].data[topIndex:topIndex+numSamples,0,0,0] = scale
            top[1].data[topIndex:topIndex+numSamples,0,0,1] = indices
            top[1].data[topIndex:topIndex+numSamples,0,0,2] = i
            topIndex+=numSamples
        assert topIndex==self.numberOfPairs, str(topIndex)+' '+str(self.numberOfPairs)
        
    def backward(self, top, propagate_down, bottom):
        if not propagate_down[0]:
            return
        topGrad = top[0].diff[...]
        bottomGrad = np.zeros_like(bottom[0].data)
        topdata = top[1].data[...]
        
        spxId2s = topdata[:,0,0,1]
        diffsSpx2s = topGrad[:,0,0,:-1]
        for i in range(self.numberOfPairs):
            bottomGrad[0,0,int(spxId2s[i]),:]+=diffsSpx2s[i]
        bottom[0].diff[...]=bottomGrad
        bottom[0].reshape(*bottomGrad.shape)

