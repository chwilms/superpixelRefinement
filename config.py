'''
Modified version of the original code from Hu et al., CVPR 2017

@author Hu et al.
@author Christian Wilms
@date 01/05/21
'''
RGB_MEAN = [123.68, 116.779, 103.939]
IMAGE_PATH_FORMAT = "data/coco/%s/%s"
IMAGE_NAME_FORMAT = "COCO_%s_%012d.jpg"
IMAGE_SET = "train2014"
ANNOTATION_TYPE = "train2014"
ANNOTATION_FILE_FORMAT = "data/coco/annotations/instances_%s.json"
SLIDING_WINDOW_SIZE = 10
MASK_SIZE = 40
BOX_SELECTION_PROCESS = 10
