'''
With code from Hu et al., CVPR2017

@author Hu et al.
@author Christian Wilms
@date 01/05/21
'''

import os
import argparse
import time
import cjson

from alchemy.utils.mask import encode
from alchemy.utils.load_config import load_config
from alchemy.utils.progress_bar import printProgress

import config
import numpy as np

import multiprocessing
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.io import imread
import itertools
import cv2 
import math

DISK = disk(1)

def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('model', type=str)
    parser.add_argument('--dataset', dest='dataset', type=str, 
                        default='val2017LVIS')
    parser.add_argument('--end', dest='end', type=int, default=5000)
    parser.add_argument('--num_workers', dest='numWorkers', type=int, default=6)

    args = parser.parse_args()
    return args

def getNeighbours(adjacencyMatrix, seedSpxId, level, maxLevel):
    if level >= maxLevel:
        return set()
    neighbours = set()
    neighbours.update(np.nonzero(adjacencyMatrix[seedSpxId,:])[0])
    for spxId in list(neighbours):
        neighbours.update(getNeighbours(adjacencyMatrix, spxId, level+1, maxLevel))
    return neighbours

def gaussian(x, sigma):
    return 1.0/(math.sqrt(2.0*math.pi)*sigma)*math.exp((-float(x)**2)/(2.0*sigma**2))
        
def colorDistFilter(values,distsColor,sigmaColor):
    #first values are always values of the central supeprixel
    W = 0.0
    S = 0.0
    for v,dC in zip(values,distsColor):
        weight = gaussian(dC,sigmaColor)
        W+=weight
        S+= weight*v
    return (1/W)*S

def f(img_id):
    if not os.path.exists('./intermediateResults/image_'+str(img_id)+'.npz'):
        return []
    loaded = np.load('./intermediateResults/image_'+str(img_id)+'.npz')
    img = imread('./data/coco/val2017LVIS/COCO_val2017LVIS_'+str(img_id).zfill(12)+'.jpg').astype(np.float)/255.0
    if len(img.shape) == 2:
        img = np.dstack([img]*3)

    img = cv2.resize(img.astype(np.float), (loaded['seg8'].shape[1],loaded['seg8'].shape[0]))

    #load segmentations from intermediate results
    segmentation = {}
    segmentation[8] = loaded['seg8'].astype(np.int)
    segmentation[16] = loaded['seg16'].astype(np.int)
    segmentation[24] = loaded['seg24'].astype(np.int)
    segmentation[32] = loaded['seg32'].astype(np.int)
    segmentation[48] = loaded['seg48'].astype(np.int)
    segmentation[64] = loaded['seg64'].astype(np.int)
    segmentation[96] = loaded['seg96'].astype(np.int)
    segmentation[128] = loaded['seg128'].astype(np.int)

    aMs = {} #creating adjacency matrix per segmentation
    for scale in [8,16,24,32,48,64,96,128]:
        aM = np.zeros([segmentation[scale].max() + 1]*2)
        aM[segmentation[scale][:, :-1], segmentation[scale][:, 1:]] = 1
        aM[segmentation[scale][:, 1:], segmentation[scale][:, :-1]] = 1
        aM[segmentation[scale][:-1, :], segmentation[scale][1:, :]] = 1
        aM[segmentation[scale][1:, :], segmentation[scale][:-1, :]] = 1
        aM[range(aM.shape[0]),range(aM.shape[0])]=0
        aMs[scale] = aM

    Is = {8:{}, 16:{}, 24:{}, 32:{}, 48:{}, 64:{}, 96:{}, 128:{}}

    #load other data from intermediate results
    oh, ow = loaded['outShape']
    objnBlob = loaded['objn']
    top_kBlob = loaded['top_k']
    obj_indicesBlob = loaded['obj_indices']
    batchSpxInfosBlob = loaded['batchSpxInfos']
    spx_score_sigBlob = loaded['spx_score_sig']

    dynamicK = 1000
    if len(objnBlob) < dynamicK:
        dynamicK = len(objnBlob.data)
    #determine dynamically how many windows are sampled at test time
    #might be less than 1000

    #extract objectness scores
    ret_scores = np.zeros((dynamicK))    
    _ = 0
    for topk in top_kBlob[:,0,0,0][:dynamicK]:
        if topk < obj_indicesBlob.shape[0]:
            score = float(objnBlob[int(topk)])
            ret_scores[_] = score
            _+=1
            
    infos = batchSpxInfosBlob.astype(np.int)
    pairScores = spx_score_sigBlob[:,0,0,0]

    numSegs = infos[:,0,0,2]+1
    numSegsIds = np.nonzero(numSegs)[0]
    infos = infos[numSegsIds,0,0,:] 
    numSegs = infos[:,2]
    scales = infos[:,0]
    sampleIds = scales*1000+numSegs
    uniqueNumSegs, uniqueNumSegsIds = np.unique(sampleIds, return_index=True)
    
    masks = []
    coordsCache={}

    sigmaColor = 0.08
    UPPER = 0.4
    LOWER = 0.25

    for infosSlice, pairScores in zip(np.split(infos, uniqueNumSegsIds[1:]),np.split(pairScores, uniqueNumSegsIds[1:])):
        Ps={}
        weak=[]
        scale = infosSlice[0,0]
        spxIds = infosSlice[:,1]
        mask = np.zeros_like(segmentation[scale], dtype=np.float)
        for i, spxId in enumerate(spxIds):
            if not coordsCache.has_key((scale,spxId)):
                coordsCache[(scale,spxId)]=segmentation[scale]==spxId
            mask[coordsCache[(scale,spxId)]] = pairScores[i]
            Ps[spxId]=pairScores[i]
            if pairScores[i] > UPPER:
                pass
            elif pairScores[i] > LOWER:
                weak.append(spxId)
        
        for centerSpx in weak:
            neighbours = getNeighbours(aMs[scale], centerSpx, 0, 2)
            if not Is[scale].has_key(centerSpx):
                if coordsCache.has_key((scale,centerSpx)):
                    xs,ys = np.where(coordsCache[(scale,centerSpx)])
                else:
                    xs,ys = np.where(segmentation[scale]==centerSpx)
                if len(xs)==0:
                    Is[scale][centerSpx]=0
                else:
                    Is[scale][centerSpx]=np.mean(img[xs,ys,:], axis = 0)
            centerMean = Is[scale][centerSpx]
            if Ps.has_key(centerSpx):
                centerP = Ps[centerSpx]
            else:
                centerP = 0

            distsColor = [0]
            distsMask = [0]
            values = [centerP]
            for nSpx in neighbours:
                if not Is[scale].has_key(nSpx):
                    if coordsCache.has_key((scale,nSpx)):
                        xs,ys = np.where(coordsCache[(scale,nSpx)])
                    else:
                        xs,ys = np.where(segmentation[scale]==nSpx)
                    if len(xs)==0:
                        Is[scale][nSpx]=0
                    else:
                        Is[scale][nSpx]=np.mean(img[xs,ys,:], axis = 0)
                nMean = Is[scale][nSpx]
                diff = np.min(np.sum((np.array(centerMean)-np.array(nMean))**2)**.5)
                distsColor.append(diff)
                if Ps.has_key(nSpx):
                    values.append(Ps[nSpx])
                else:
                    values.append(0)
                distsMask.append(abs(values[0]-values[-1]))

            newP = colorDistFilter(values,distsColor,sigmaColor)
            xs,ys = np.where(segmentation[scale]==centerSpx)
            priorSum = np.sum(mask)
            mask[xs,ys]=newP
            
        mask = cv2.resize(mask, (ow, oh))
         
        #post-processing
        mask = binary_opening(mask>.3, DISK)
        mask = binary_closing(mask>.3, DISK)
        masks.append(mask)
            
    masks = np.array(masks)
    
    assert len(masks) ==len(ret_scores), (len(masks),len(ret_scores))
    
    #create list of local results (one image)
    loc_results = []
    for _ in range(len(masks)):
        score = float(ret_scores[_])
        objn = float(ret_scores[_])
        loc_results.append({
            'my_id': img_id*1000+_,
            'image_id': img_id,
            'category_id': 1, #as we are doing class-agnostic proposal 
                              #generation, cat_id is irrelevant
            'segmentation': encode(masks[_]),
            'score': score,
            'objn': objn
            })
    return loc_results

if __name__ == '__main__':
    args = parse_args()

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

    #generate all image IDs
    imageIds = []
    for i in range(len(ds)):
        batch = spider.fetch()
        image_id = int(ds[i].image_path[-16:-4])
        imageIds.append(image_id)
        
    #generate finale results in parallel
    results = []
    p = multiprocessing.Pool(args.numWorkers)
    allLocResults = p.map(f, imageIds)
    results=list(itertools.chain(*allLocResults))
    p.close()   

    with open('results/%s.json' % args.model, "wb") as f:
        f.write(cjson.encode(results))

