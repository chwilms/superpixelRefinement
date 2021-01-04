#include "caffe/layers/scale_flags_layer.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include <typeinfo>

//#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void FWD(const int nthreads, const Dtype* inputFlags,
    const int scalefactor, const int height, const int width, const int height_seg, const int width_seg, Dtype* outputFlags
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int h_flag = index / (width+1);
    int w_flag = index % (width+1);
    h_flag = min(height_seg, h_flag*scalefactor);
    w_flag = min(width_seg, w_flag*scalefactor);

    outputFlags[h_flag*(width_seg+1)+w_flag] = inputFlags[index];
  }
}

template <typename Dtype>
void ScaleFlagsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* outputFlags = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(),(Dtype) 0, outputFlags);

  FWD<Dtype> <<<CAFFE_GET_BLOCKS(bottom[2]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[2]->count(), bottom[2]->gpu_data(), scalefactor_, height_, width_,height_seg_, width_seg_,outputFlags);
}


template <typename Dtype>
void ScaleFlagsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
}


INSTANTIATE_LAYER_GPU_FUNCS(ScaleFlagsLayer);

}  // namespace caffe

