#include "caffe/layers/sliding_window_index_seg_layer.hpp"
#include <cstdio>
#include <iostream>
#include <ctime>

namespace caffe {

template <typename Dtype>
__global__ void SlidingWindowIndexSegForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_flags,
    const int channels, const int bottom_height, const int bottom_width, 
    const int top_height, const int top_width, const int stride_h, const int stride_w, 
    const int window_h, const int window_w, const int numResults, Dtype* top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int n = index / channels; 
    if (bottom_flags[n] > 0){ 
      int origin_h = n / top_width;
      int origin_w = n % top_width;

      int windowIndex = 0;
      for (int i = 0; i < n; ++i){
          windowIndex += bottom_flags[i]; 
      }
      for (int h = -window_h/2; h < window_h/2; ++h)
        for (int w = -window_w/2; w < window_w/2; ++w) {
          int top_idx = windowIndex * channels * window_h * window_w +
                    c * window_h * window_w +
                    (h + window_h/2) * window_w +
                    (w + window_w/2);
          
          if (origin_h + h >= 0 && origin_h + h < bottom_height &&
              origin_w + w >= 0 && origin_w + w < bottom_width) {
            int bottom_idx =  c * bottom_height * bottom_width +
                          (origin_h + h) * bottom_width +
                          (origin_w + w);
            top_data[top_idx] = bottom_data[bottom_idx];
          }
          else
            top_data[top_idx] = -1;
        }
    }
}
  
}


template <typename Dtype>
void SlidingWindowIndexSegLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_flags = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top_height_ * top_width_ * channels_;
  SlidingWindowIndexSegForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (
      count, bottom_data, bottom_flags, channels_, bottom_height_, bottom_width_, top_height_, 
      top_width_, stride_h_, stride_w_, window_h_, window_w_, numResults_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SlidingWindowIndexSegLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(SlidingWindowIndexSegLayer);


}
