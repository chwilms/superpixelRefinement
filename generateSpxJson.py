'''
@author Christian Wilms
@date 01/05/21
'''

import os
import json
import io
import numpy as np
from skimage.transform import resize
from alchemy.datasets.coco import COCO_DS
from multiprocessing import Pool

try:
    to_unicode = unicode
except NameError:
    to_unicode = str


def f(ann):
    annId = ann['id']
    imageID = ann['image_id']
    if os.path.exists('./segmentations/train2014/fh-8-8000/'+str(imageID)+'.csv'):
        #generate gt mask from annotation
        orgGtMask = dataset.annToMask(ann)
        if np.sum(orgGtMask)==0:
            y,x,w,h = np.round(ann['bbox']).astype(np.int)
            w = max(w,1)
            h = max(h,1)
            orgGtMask[x:x+h,y:y+w]=1
        annResultDict={}
        for scale, numSpx in [(8,8000),(16,4000),(24,3000),(32,2000),(48,1500),(64,1000),(96,750),(128,500)]:
            #load the segmentation belonging to the annotation/image
            try:
                seg = np.loadtxt('./segmentations/train2014/fh-'+str(scale)+'-'+str(numSpx)+'/'+str(imageID)+'.csv', delimiter=',')
            except ValueError:
                try:
                    seg = np.loadtxt('./segmentatons/train2014/fh-'+str(scale)+'-'+str(numSpx)+'/'+str(imageID)+'.csv', delimiter=',')
                except ValueError:
                    print 'Error: Failed to load: ./segmentations/train2014/fh-'+str(scale)+'-'+str(numSpx)+'/'+str(imageID)+'.csv' 
                    
            #resize the annotaton to the size of the segmentation
            gtMask = resize(orgGtMask, seg.shape)>=.5
            if np.sum(gtMask)==0:
                gtMask = resize(orgGtMask, seg.shape)>0    
                    
            #calculate the intersection and union of all superpixels covering the annotation with the annotation
            spxIds = np.unique(seg[gtMask==1])
            assert len(spxIds) > 0, [np.sum(gtMask), np.sum(orgGtMask), np.unique(gtMask), np.sum(seg)]
            intersections = []
            unionsWOGT = []
            ious = []
            gtSize = np.sum(gtMask)
            for i in spxIds:
                spxMaskI = seg == i
                intersections.append(np.sum(np.logical_and(gtMask,spxMaskI)))
                union = np.sum(np.logical_or(gtMask,spxMaskI))
                unionsWOGT.append(max(union-gtSize, 0.000001))
                ious.append(float(intersections[-1])/union)

            #seed superpixel is the one with the highest IoU
            bestFittingSpx =np.argmax(ious) 
            startIoU = ious[bestFittingSpx]
            intersections = np.array(intersections)
            unionsWOGT = np.array(unionsWOGT)
            
            #calculate the gain w.r.t. the initial IoU of each superpixel
            gain = intersections/unionsWOGT
    
            #add all superpixels that increase the IoU with the annotation
            #they ultimately form the gt
            ids = np.nonzero(gain>=startIoU)[0]
            if not bestFittingSpx in ids:
                ids = ids.tolist()
                ids.append(bestFittingSpx)
                ids = np.array(ids)
            result = spxIds[ids]
            annResultDict[scale]=result.tolist()
        return (annId,annResultDict)
    else:
        return ()
    
dataset = COCO_DS('./data/coco/annotations/instances_train2014.json', True)

#parallel processing of the annotations
pool = Pool(processes=6) 
result = pool.map(f, dataset.anns.values())
result = filter(lambda x:x!=(), result)
print len(result)
overallResultDict = dict(result)

with io.open('spxGT_train2014_FH.json', 'w', encoding='utf-8') as f:
    str_ = json.dumps(overallResultDict, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
    f.write(to_unicode(str_))

