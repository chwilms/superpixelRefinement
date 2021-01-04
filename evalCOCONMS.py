'''
Modified version of the original code from Hu et al., CVPR 2017

@author Hu et al.
@author Christian Wilms
@date 01/05/21
'''

import argparse
import cjson
import config
from config import *

from pycocotools.cocoeval import COCOeval

from alchemy.utils.mask import iou

import multiprocessing
import itertools
from collections import defaultdict
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('model', type=str)
    parser.add_argument('--useSegm', dest='useSegm', type=str, default='True')
    parser.add_argument('--end', dest='end', type=int, default=5000)
    parser.add_argument('--nms_threshold', dest='nms_threshold', type=float, default=0.95)
    parser.add_argument('--dataset', dest='dataset', type=str, default='val2017LVIS')
    parser.add_argument('--max_proposal', dest='max_proposal', type=int, default=1000)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=6)

    args = parser.parse_args()
    args.useSegm = args.useSegm == 'True'
    return args
    
def func(sub):
    sub.sort(key=lambda item: item['objn'], reverse=True)
    # nms
    keep = np.ones(len(sub)).astype(np.bool)
    if args.nms_threshold < 1:
        for i in range(len(sub)):
            if keep[i]:
                for j in range(i+1, len(sub)):
                    if keep[j] and iou(sub[i]['segmentation'], sub[j]['segmentation'], [False]) > args.nms_threshold:
                        keep[j] = False


    for i in reversed(np.where(keep==False)[0]):
        del sub[i]
    sub = sub[:args.max_proposal]

    return sub

if __name__ == '__main__':
    args = parse_args()

    max_dets = [1, 10, 100, 1000]
    i = 1
    
    with open('results/%s.json' % args.model, 'rb') as f:
        input_results = cjson.decode(f.read())
        results = []
        _ = 0
        overallSubResults = defaultdict(list)
        while _ < len(input_results):
            overallSubResults[input_results[_]['image_id']].append(input_results[_])
            _+=1
         
        #NMS in parallel
        p = multiprocessing.Pool(args.num_workers)
        allLocResults = p.map(func, overallSubResults.values())
        results=list(itertools.chain(*allLocResults))
        p.close()

    with open("results/%s_temp.json" % args.model, 'wb') as f:
        f.write(cjson.encode(results))
    
    config.ANNOTATION_TYPE = args.dataset
    config.IMAGE_SET = args.dataset
    from spiders.coco_ssm_spider import COCOSSMDemoSpiderSeg
    spider = COCOSSMDemoSpiderSeg()
    ds = spider.dataset

    cocoGt = ds
    cocoDt = cocoGt.loadRes("results/%s_temp.json" % args.model)
    cocoEval = COCOeval(cocoGt, cocoDt)

    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())[:args.end]
    cocoEval.params.maxDets = max_dets
    cocoEval.params.useSegm = args.useSegm
    cocoEval.params.useCats = False
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

