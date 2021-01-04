'''
Modified version of the original code from Hu et al., CVPR 2017

@author Hu et al.
@author Christian Wilms
@date 01/05/21
'''

from spiders.coco_ssm_spider import *
from alchemy.engines.caffe_python_layers import AlchemyDataLayer

class COCOSSMSpiderAttentionMask8_128(AlchemyDataLayer):

    spider =  COCOSSMSpiderAttentionMask8_128
    
