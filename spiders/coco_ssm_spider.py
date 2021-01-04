'''
Modified version of the original code from Hu et al., CVPR 2017

@author Hu et al.
@author Christian Wilms
@date 01/05/21
'''

from config import *
import numpy as np
from alchemy.datasets.coco import COCO_DS
from skimage.transform import resize

from base_coco_ssm_spider import *

class COCOSSMSpiderAttentionMask8_128(BaseCOCOSSMSpiderAttentionMask):

    attr = ['image', 'objAttMask_8', 'objAttMask_16', 'objAttMask_24', 'objAttMask_32', 'objAttMask_48', 'objAttMask_64', 'objAttMask_96', 'objAttMask_128', 'objAttMask_8_org', 'objAttMask_16_org', 'objAttMask_24_org', 'objAttMask_32_org', 'objAttMask_48_org', 'objAttMask_64_org', 'objAttMask_96_org', 'objAttMask_128_org']

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, True)
            self.__class__.cats_to_labels = dict([(self.dataset.getCatIds()[i], i+1) for i in range(len(self.dataset.getCatIds()))])
        super(COCOSSMSpiderAttentionMask8_128, self).__init__(*args, **kwargs)
        self.RFs = RFs
        self.SCALE = SCALE

class COCOSSMDemoSpiderSeg(BaseCOCOSSMSpiderAttSizeTest):

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, False)
        super(COCOSSMDemoSpiderSeg, self).__init__(*args, **kwargs)
        try:
            self.RFs = RFs
        except Exception:
            pass
        try:
            self.SCALE = TEST_SCALE
        except Exception:
            pass

    def fetch(self, skip=False):
        idx = self.get_idx()
        if skip:
            return
        item = self.dataset[idx]
        self.image_path = item.image_path
        self.id = int(self.image_path.split('_')[-1].split('.')[0])
        self.anns = item.imgToAnns
        self.max_edge = self.SCALE
        self.fetch_image()
        result = {"image": self.img_blob}

        scaleNSpxs = dict(zip(RFs,SPXs))
        for i in self.RFs:
            seg = np.loadtxt('./segmentations/val2017LVIS/fh-'+str(i)+'-'+str(scaleNSpxs[i])+'/'+str(self.id)+'.csv', delimiter=',')
            result['seg_'+str(i)]=np.expand_dims(np.expand_dims(seg,0),0)
        return result

