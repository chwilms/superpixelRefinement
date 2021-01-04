'''
Modified version of the original code from Hu et al., CVPR 2017

@author Hu et al.
@author Christian Wilms
@date 01/05/21
'''

from __future__ import division
import numpy as np

from alchemy.utils.image import resize_blob
from alchemy.utils.mask import crop
from skimage.segmentation import relabel_from_one
import cv2

def transplant(new_net, net, suffix=''):
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

def expand_score(new_net, new_layer, net, layer):
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# generate masks from an image with specified net
# :param net:           caffe net
# :param input:         input image blob ([1, 3, h, w])
# :param config:        other parameters
# :param dest_shape:    resize masks if specified
# :param image:         visualize masks if specified
# :return masks:        masks ([num, h, w])
def storeIntermediateResults(net, input, mask8, mask16, mask24, mask32, mask48, mask64, mask96, mask128, image_id, dest_shape=None, image=None):
    net.blobs['data'].reshape(*input.shape)
    net.blobs['data'].data[...] = input
    
    net.blobs['seg_8'].reshape(*mask8.shape)
    net.blobs['seg_8'].data[...] = mask8
    net.blobs['seg_16'].reshape(*mask16.shape)
    net.blobs['seg_16'].data[...] = mask16
    net.blobs['seg_24'].reshape(*mask24.shape)
    net.blobs['seg_24'].data[...] = mask24
    net.blobs['seg_32'].reshape(*mask32.shape)
    net.blobs['seg_32'].data[...] = mask32
    net.blobs['seg_48'].reshape(*mask48.shape)
    net.blobs['seg_48'].data[...] = mask48
    net.blobs['seg_64'].reshape(*mask64.shape)
    net.blobs['seg_64'].data[...] = mask64
    net.blobs['seg_96'].reshape(*mask96.shape)
    net.blobs['seg_96'].data[...] = mask96
    net.blobs['seg_128'].reshape(*mask128.shape)
    net.blobs['seg_128'].data[...] = mask128

    net.forward()

    ih, iw = input.shape[2:]
    if dest_shape != None:
        oh, ow = dest_shape
    else:
        oh, ow = ih, iw
    oh, ow = int(oh), int(ow)
    
    np.savez_compressed('./intermediateResults/image_'+str(image_id)+'.npz', seg8=mask8[0,0], seg16=mask16[0,0], seg24=mask24[0,0], seg32=mask32[0,0], seg48=mask48[0,0], seg64=mask64[0,0], seg96=mask96[0,0], seg128=mask128[0,0],
                        outShape=np.array([oh,ow]), objn=net.blobs['objn'].data[...], top_k=net.blobs['top_k'].data[...], obj_indices=net.blobs['obj_indices'].data[...],
                        batchSpxInfos=net.blobs['batchSpxInfos'].data[...], spx_score_sig=net.blobs['spx_score_sig'].data[...])
