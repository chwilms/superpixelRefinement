#include "caffe/layers/spx_pooling_layer.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include <typeinfo>

namespace caffe {

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}


template <typename Dtype>
__global__ void SUM(const int nthreads, const Dtype* featureMap,
    const Dtype* spxMap, const int channels, const int width_spx, const int height_spx, const int width_feat, const int height_feat, const float scalefactor, Dtype* spxPoolSum, Dtype* pxCount
    ) {
  int numPx = height_feat * width_feat;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int n = index / channels; 
    //n = (n+index*32)%nthreads;
    int h_s = n / width_spx;
    int w_s = n % width_spx;
    float h_s_in_f = h_s * scalefactor; //coord of spx map in feature map coords
    float hw = min(height_feat-1.0f,max(round(h_s_in_f),0.0f)) * width_feat;
    float w_s_in_f = w_s * scalefactor;
    float final_index = hw + min(width_feat-1.0f,max(round(w_s_in_f),0.0f));
    int spxId = (int)spxMap[h_s * width_spx + w_s];

    if (c == 0)
      atomicAdd(&pxCount[spxId],1);
//      pxCount[0] = pxCount[0]+1;

    atomicAdd(&spxPoolSum[spxId * channels + c],featureMap[(int) (c * numPx + final_index)]);
//    spxPoolSum[0]=1.0;//+=featureMapFloat[(int) (c * numPx + final_index)];
  }
}




template <typename Dtype>
__global__ void AVG(const int nthreads, const int channels, Dtype* spxPoolSum, Dtype* pxCount)
  {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    if (pxCount[n] > 0)
      spxPoolSum[index] = spxPoolSum[index]/pxCount[n];
  }
}



template <typename Dtype>
void SpxPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[1]->count();
  Dtype* spxPoolSum = top[0]->mutable_gpu_data();
  Dtype* pxCountGPU = top[1]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(),(Dtype) 0, spxPoolSum);
  caffe_gpu_set(top[1]->count(),(Dtype) 0, pxCountGPU);


  SUM<Dtype> <<<CAFFE_GET_BLOCKS(count*channels_), CAFFE_CUDA_NUM_THREADS>>>(
        count*channels_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), channels_, width_spx_, height_spx_, width_feat_, height_feat_, scalefactor_, spxPoolSum, pxCountGPU);

  AVG<Dtype> <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), channels_, spxPoolSum, pxCountGPU);

}

template <typename Dtype>
__global__ void BACK(const int nthreads, const Dtype* top_diff,
    const Dtype* spxMap, const int channels, const int width_spx, const int width_feat, const int height_feat, const float scalefactor, const float weight, Dtype* bottom_diff, const Dtype* pxCount
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int n = index / channels; 
    int h = n / width_feat;
    int w = n % width_feat;

    int spxId = (int)spxMap[(int) (h/scalefactor * width_spx + w/scalefactor)];
    int pxC = pxCount[spxId];
    float newWeight = weight / pxC;
    bottom_diff[c * height_feat * width_feat + h * width_feat + w] = newWeight * top_diff[spxId * channels + c];

  } //CUDA_KERNEL_LOOP
}

template <typename Dtype>
void SpxPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* spxMap = bottom[1]->gpu_data();
  const Dtype* pxCount = top[1]->gpu_data();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

  float weight = this->layer_param_.superpixel_pooling_param().weight();

  BACK<Dtype> <<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), top_diff, spxMap, channels_, width_spx_, width_feat_, height_feat_, scalefactor_, weight, bottom_diff, pxCount);
  
}


template <>
void SpxPoolingLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
NOT_IMPLEMENTED;
}



__global__ void SUMDouble(const int nthreads, const double* featureMap,
    const double* spxMap, const int channels, const int width_spx, const int height_spx, const int width_feat, const int height_feat, const float scalefactor, double* spxPoolSum, double* pxCount
    ) {
  const float* featureMapFloat = (const float*) featureMap;
  int numPx = height_feat * width_feat;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int n = index / channels; 
    int h_s = n / width_spx;
    int w_s = n % width_spx;
    float h_s_in_f = h_s * scalefactor; //coord of spx map in feature map coords
    float hw = min(height_feat-1.0f,max(round(h_s_in_f),0.0f)) * width_feat;
    float w_s_in_f = w_s * scalefactor;
    float final_index = hw + min(width_feat-1.0f,max(round(w_s_in_f),0.0f));
    int spxId = (int)spxMap[h_s * width_spx + w_s];

    if (c == 0)
      atomicAddDouble(&pxCount[spxId],1);

    atomicAddDouble(&spxPoolSum[spxId * channels + c],featureMapFloat[(int) (c * numPx + final_index)]);
  }
}

__global__ void AVGDouble(const int nthreads, const int channels, double* spxPoolSum, double* pxCount)
  {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    spxPoolSum[index] = spxPoolSum[index]/pxCount[n];
  }
}


template <>
void SpxPoolingLayer<double>::Forward_gpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  const int count = bottom[1]->count();
  double* spxPoolSum = top[0]->mutable_gpu_data();
  double* pxCountGPU = top[1]->mutable_gpu_data();
//  unsigned int pxCountGPU[(int) (maxValSeg_+1)];
//  std::memset(pxCountGPU, 0, sizeof pxCountGPU);

  std::cout<<"hello from GPUDouble++++++++++++++++++++++++++++++++++++ "<<typeid(top[0]->mutable_gpu_data()).name()<<"\n";

  SUMDouble <<<CAFFE_GET_BLOCKS(count)*channels_, CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), channels_, width_spx_, height_spx_, width_feat_, height_feat_, scalefactor_, spxPoolSum, pxCountGPU);

  std::cout<<"hello from GPU1Double++++++++++++++++++++++++++++++++++++\n";
  AVGDouble <<<CAFFE_GET_BLOCKS(maxValSeg_+1), CAFFE_CUDA_NUM_THREADS>>>(
        (int) (maxValSeg_+1), channels_, spxPoolSum, pxCountGPU);
  std::cout<<"hello from GPU2Double++++++++++++++++++++++++++++++++++++\n";

}

INSTANTIATE_LAYER_GPU_FUNCS(SpxPoolingLayer);

}  // namespace caffe

