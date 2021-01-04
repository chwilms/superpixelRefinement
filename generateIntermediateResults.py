'''
With code from Hu et al. CVPR 2017

@author Hu et al.
@author Christian Wilms
@date 01/05/21
'''

import sys
import os
import argparse
import time
import cjson
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
import caffe
import setproctitle 

from alchemy.utils.mask import encode
from alchemy.utils.load_config import load_config
from alchemy.utils.progress_bar import printProgress

import config
import utils
from config import *
import numpy as np

from utils import storeIntermediateResults


def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('--init_weights', dest='init_weights', type=str,
                        default=None)
    parser.add_argument('--dataset', dest='dataset', type=str, 
                        default='val2017LVIS')
    parser.add_argument('--end', dest='end', type=int, default=5000)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))
    setproctitle.setproctitle(args.model)

    net = caffe.Net(
            'models/' + args.model + ".test.prototxt",
            'params/' + args.init_weights,
            caffe.TEST)

    # surgeries
    interp_layers = [layer for layer in net.params.keys() if 'up' in layer]
    utils.interp(net, interp_layers)

    if os.path.exists("configs/%s.json" % args.model):
        load_config("configs/%s.json" % args.model)
    else:
        print "Specified config does not exists, use the default config..."
        
    time.sleep(2)

    config.ANNOTATION_TYPE = args.dataset
    config.IMAGE_SET = args.dataset
    from spiders.coco_ssm_spider import COCOSSMDemoSpiderSeg
    spider = COCOSSMDemoSpiderSeg()
    spider.dataset.sort(key=lambda item: int(item.image_path[-16:-4]))
    ds = spider.dataset[:args.end]

    results = []
    for i in range(len(ds)):
        if i < 0: #set for continue testing with the nth image
            batch = spider.fetch(True)
            continue
        else:
            batch = spider.fetch()
        img = batch["image"]
        image_id = int(ds[i].image_path[-16:-4])
        mask8 = batch["seg_8"]
        mask16 = batch["seg_16"]
        mask24 = batch["seg_24"]
        mask32 = batch["seg_32"]
        mask48 = batch["seg_48"]
        mask64 = batch["seg_64"]
        mask96 = batch["seg_96"]
        mask128 = batch["seg_128"]

#        print i, image_id
        if image_id in [131431,304545]:
            continue
 
        storeIntermediateResults(net, img, mask8, mask16, mask24, mask32, mask48, mask64, mask96, mask128, image_id, 
                dest_shape=(spider.origin_height, spider.origin_width)) 

        printProgress(i, len(ds), prefix='Progress: ', suffix='Complete', barLength=50)

