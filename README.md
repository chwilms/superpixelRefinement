# Superpixel-based Refinement for Object Proposal Generation
[Superpixel-based Refinement for Object Proposal Generation (ICPR 2020)](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people-alt/wilms/spxrefinement.html)

Precise segmentation of objects is an important problem in tasks like class-agnostic object proposal generation or instance segmentation. Deep learning-based systems usually generate segmentations of objects based on coarse feature maps, due to the inherent downsampling in CNNs. This leads to segmentation boundaries not adhering well to the object boundaries in the image. To tackle this problem, we introduce a new superpixel-based refinement approach on top of the state-of-the-art object proposal system AttentionMask. The refinement utilizes superpixel pooling for feature extraction and a novel superpixel classifier to determine if a high precision superpixel belongs to an object or not. Our experiments show an improvement of up to 26.0% in terms of average recall compared to original AttentionMask. Furthermore, qualitative and quantitative analyses of the segmentations reveal significant improvements in terms of boundary adherence for the proposed refinement compared to various deep learning-based state-of-the-art object proposal generation systems.

![Example](/example.png)

The system is based on [AttentionMask](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/wilms/attentionmask.html) and [FastMask](https://arxiv.org/abs/1612.08843).

If you find this software useful in your research, please cite our paper.

```
@inproceedings{WilmsFrintropICPR2020,
 title = {{Superpixel-based Refinement for Object Proposal Generation}, 
 author = {Christian Wilms and Simone Frintrop},
 booktitle = {International Conference on Pattern Recognition (ICPR)},
 year = {2020}
}
```
# Requirements
- Ubuntu 18.04 
- Cuda 10.0
- Python 2.7
- OpenCV-Python
- Python packages: scipy, numpy, python-cjson, setproctitle, scikit-image
- [COCOApi](https://github.com/pdollar/coco)
- Caffe (already part of this git)
- [Alchemy](https://github.com/voidrank/alchemy) (already part of this git)

# Hardware specifications
For the results in the paper we used the following hardware:
- Intel i7-5930K 6 core CPU
- 64 GB RAM
- GTX Titan X GPU with 12 GB RAM

# Installation
Follow the installation instructions in the [AttentionMask git](https://github.com/chwilms/AttentionMask#installation)

# Usage
Our superpixel-based refinement system can be used without any re-training, utilizing our provided weights and segmentation. Just download the weights,the LVIS dataset and the segmentations.

## Download dataset
Download the `train2014` splits from [COCO dataset](http://cocodataset.org/#download) for training and the validation split form the [LVIS dataset](https://www.lvisdataset.org/dataset). After downloading, extract the data in the following structure:

```
spxattmask
|
---- data
     |
     ---- coco
          |
          ---- annotations
          |    |
          |    ---- instances_train2014.json
          |    |
          |    ---- instances_val2017LVIS.json
          |
          ---- train2014
          |    |
          |    ---- COCO_train2014_000000000009.jpg
          |    |
          |    ---- ...
          |
          ---- val2017LVIS
               |
               ---- COCO_val2017LVIS_000000000139.jpg
               |
               ---- ...
```

## Download weights
Download our weights for the superpixel-based refinement system: [Link to caffemodel](https://fiona.uni-hamburg.de/a3c1f3ec/spxrefinedattmask-final.caffemodel).

For training the system on your own dataset, download the [initial ImageNet weights for the ResNet-34](https://fiona.uni-hamburg.de/a3c1f3ec/resnet34.caffemodel). 

All weight files (.caffemodel) should be moved into the `params` subdirectory.

## Creating segmentations
Essential to our superpixel-based refinement system are the superpixel segmentations. We generated and optimized all superpixel segmentations using the [framework by Stutz et al](https://github.com/davidstutz/superpixel-benchmark). For training and testing eight segmentations needs to be generated per image, one segmentation per AttentionMask scale. 

Due to the size we will not provide the segmentations. However, the segmentations used in the paper can be reproduced  using the [framework by Stutz et al](https://github.com/davidstutz/superpixel-benchmark). Follow the instalation insturctions in that repo. Note that only the segmentation algorithm by Felzenszwalb and Huttenlocher has to be build. The following table provides the parameters (`scale (-t)`, `minimum-size (-m)`, `sigma (-g)` in the framework by Stutz et al.) for generating the segmentations for each of the eight scales in AttentionMask.

| Scale        | Parameter `scale (-t)` | Parameter `minimum-size (-m)` | Parameter `sigma (-g)` |
| ------------- |:-------------:|:-----:|:-----:|
| 8 | 10 | 10 | 1 |
| 16 | 60 | 15 | 0 |
| 24 | 60 | 30 | 0 |
| 32 | 120 | 30 | 0 |
| 48 | 10 | 60 | 1 |
| 64 | 30 | 90 | 1 |
| 96 | 60 | 120 | 1 |
| 128 | 10 | 180 | 0 |

The segmentation size, i.e., the image size during segmentation, as well information about flipping the image and the segmentation (training only) can be found in the following json files for [training data](https://fbicloud.informatik.uni-hamburg.de/index.php/s/xoRpzTWxcDB9NWM) and [test data](https://fbicloud.informatik.uni-hamburg.de/index.php/s/jgcTeZzeqRFbDLx). The json files contain a mapping from the image id to the height and width of the segmentation as well as a flag for denoting a left-right-flip (training only).

### Segmentations for training
For training, the segmentations have to be provided in two different ways. First, all segmentations are expected as compressed csv-file (``csv.gz``) in the subdirectory ``segmentations/train2014/`` with an indidividual folder per scale. Additionally, from those segmentations the superpixelized ground truth needs to be generated as ``json``-file with scipt ``generateSpxJson.py`` followed by the script ``splitJson.py``. Generate the segmentations using the [framework by Stutz et al.](https://github.com/davidstutz/superpixel-benchmark) and the parameters discussed above and  paste the results into the directory structure shown below.

### Segmentations for testing
During testing, only the superpixel segmentations are necessary. The segmentations are expeted as ``csv``-file in the subdirectory ``segmentations/val2017LVIS`` with an individual folder per scale. Generate the segmentations using the [framework by Stutz et al.](https://github.com/davidstutz/superpixel-benchmark) and the parameters discussed above and paste the results into the directory structure shown below.

```
spxattmask
|
---- spxGT_train2014_FH_128.json
|
---- spxGT_train2014_FH_16.json
|
---- spxGT_train2014_FH_24.json
|
---- spxGT_train2014_FH_32.json
|
---- spxGT_train2014_FH_48.json
|
---- spxGT_train2014_FH_64.json
|
---- spxGT_train2014_FH_8.json
|
---- spxGT_train2014_FH_96.json
|
---- segmentations
     |
     ---- train2014
     |    |
     |    ---- fh-8-8000
     |    |    |
     |    |    ---- 132574.csv.gz
     |    |    |
     |    |    ---- ...
     |    |
     |    ---- ...
     |
     ---- val2017LVIS
          |
          ---- fh-8-8000
          |    |
          |    ---- 1000.csv
          |    |
          |    ---- ...
          |
          ---- ...
```

## Inference
For inference on the LVIS dataset, first use the script `generateIntermediateResults.py` that runs the images thorugh the CNN and generates intermediate reuslts. Those results are stored in the folder `intermediateResults`, which has to be created first. Call the script with the gpu id, the model name, the weights and the dataset you want to test on (e.g., val2017LVIS):

```
$ python generateIntermediateResults.py 0 spxRefinedAttMask --init_weights spxrefinedattmask-final.caffemodel --dataset val2017LVIS --end 5000
```

Second, to apply the post-processing to the results and to stitch the proposals back into the image, call `generateFinalResults.py` with the model name and the dataset:

```
$ python generateFinalResults.py spxRefinedAttMask --dataset val2017LVIS --end 5000
```

You can find an example for both calls as well as the evaluation (see below) in the script `test.sh`.

## Evaluation
Use `evalCOCONMS.py` to evaluate on the LVIS dataset with the model name and the dataset used. `--useSegm` is a flag for using segmentation masks instead of bounding boxes.

```
$ python evalCOCONMS.py spxRefinedAttMask --dataset val2017LVIS --useSegm True --end 5000
```

## Training
To train our superpixel-based refinement system on the COCO dataset, you can use the `train.sh` script. The training runs for 13 epochs to generate the final model weights.

```
$ export EPOCH=1
$ ./train.sh
```
