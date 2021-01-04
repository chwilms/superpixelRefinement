#include "caffe/layers/att_spx_pooling_layer.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include <typeinfo>
#include <stdlib.h>

namespace caffe {


template <typename Dtype>
__global__ void SUMAtt(const int nthreads, const Dtype* featureMap,
    const Dtype* spxMap, const int maxValSeg, const int batchEntries, const int width, const int height, Dtype* spxPoolSum, Dtype* pxCount
    ) {
  int numPx = height * width;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int b = index / (height * width);
    int n = index % (height * width);
    int h = n / width;
    int w = n % width;

    float spxId = spxMap[b * numPx + h * width + w];

    if (spxId >= 0){
      atomicAdd(&pxCount[(maxValSeg+1) * b + (int)spxId],1);
      atomicAdd(&spxPoolSum[(maxValSeg+1) * b + (int)spxId],featureMap[(int) (b * numPx + h * width + w)]);
    }
  }
}

template <typename Dtype>
__global__ void AVGAtt(const int nthreads, const int batch, Dtype* spxPoolSum, Dtype* pxCount)
  {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (pxCount[index] > 0){
        spxPoolSum[index] = (spxPoolSum[index]+100)/pxCount[index];
    }
  }
}

template <typename Dtype>
void AttSpxPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[1]->count();
  Dtype* spxPoolSum = top[0]->mutable_gpu_data();
  Dtype* pxCountGPU = top[1]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(),(Dtype) -100, spxPoolSum);
  caffe_gpu_set(top[1]->count(),(Dtype) 0, pxCountGPU);

  SUMAtt<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), maxValSeg_, batchEntries_, width_, height_, spxPoolSum, pxCountGPU);
  AVGAtt<Dtype> <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), batchEntries_, spxPoolSum, pxCountGPU);

}

template <typename Dtype>
__global__ void BACK(const int nthreads, const Dtype* top_diff,
    const Dtype* spxMap, const int maxValSeg, const int batchEntries, const int width, const int height, Dtype* bottom_diff, const Dtype* pxCount
    ) {
  int numPx = height * width;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int b = index / (height * width);
    int n = index % (height * width);
    int h = n / width;
    int w = n % width;

    int spxId = (int)spxMap[b * numPx + h * width + w];
    int pxC = pxCount[spxId];
    if (pxC <= 0)
	bottom_diff[b * numPx + h * width + w] = 0;
    else
        bottom_diff[b * numPx + h * width + w] = 0;
  
  } //CUDA_KERNEL_LOOP
}

template <typename Dtype>
void AttSpxPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff1 = bottom[1]->mutable_gpu_diff();
  Dtype* bottom_diff2 = bottom[2]->mutable_gpu_diff();
  const Dtype* spxMap = bottom[1]->gpu_data();
  const Dtype* pxCount = top[1]->gpu_data();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom_diff1);
  caffe_gpu_set(bottom[2]->count(), Dtype(0), bottom_diff2);
  
  BACK<Dtype> <<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), top_diff, spxMap, maxValSeg_, batchEntries_, width_, height_, bottom_diff, pxCount);
  CUDA_POST_KERNEL_CHECK;
  CHECK(bottom_diff);
}



template <>
void AttSpxPoolingLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
NOT_IMPLEMENTED;
}


template <>
void AttSpxPoolingLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(AttSpxPoolingLayer);

}  // namespace caffe

