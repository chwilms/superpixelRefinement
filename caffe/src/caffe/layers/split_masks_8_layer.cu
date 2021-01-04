#include "caffe/layers/split_masks_8_layer.hpp"
#include <vector>
#include <iostream>
#include <cstring>
#include <typeinfo>

namespace caffe {


template <typename Dtype>
__global__ void FWD(const int nthreads, const Dtype* input,
    const int channels, const int height, const int width, Dtype* masks8, Dtype* masks16, Dtype* masks24, Dtype* masks32, Dtype* masks48, Dtype* masks64, Dtype* masks96, Dtype* masks128, const int len8, const int len16, const int len24, const int len32, const int len48, const int len64, const int len96, const int len128
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / (channels*height*width); 
    if (n < len8)
        masks8[index]=input[index];
    else{
    if (n < len16+len8)
        masks16[index-len8*channels*height*width]=input[index];
    else{
    if (n < len24+len16+len8)
        masks24[index-len8*channels*height*width-len16*channels*height*width]=input[index];
    else{
    if (n < len32+len24+len16+len8)
        masks32[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width]=input[index];
    else{
    if (n < len48+len32+len24+len16+len8)
        masks48[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width]=input[index];
    else{
    if (n < len64+len48+len32+len24+len16+len8)
        masks64[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width-len48*channels*height*width]=input[index];
    else{
    if (n < len96+len64+len48+len32+len24+len16+len8)
        masks96[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width-len48*channels*height*width-len64*channels*height*width]=input[index];
    else{
        masks128[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width-len48*channels*height*width-len64*channels*height*width-len96*channels*height*width]=input[index];
    }}}}}}}
  }
}

template <typename Dtype>
void SplitMasks8Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[1]->count();
  Dtype* masks8 = top[0]->mutable_gpu_data();
  Dtype* masks16 = top[1]->mutable_gpu_data();
  Dtype* masks24 = top[2]->mutable_gpu_data();
  Dtype* masks32 = top[3]->mutable_gpu_data();
  Dtype* masks48 = top[4]->mutable_gpu_data();
  Dtype* masks64 = top[5]->mutable_gpu_data();
  Dtype* masks96 = top[6]->mutable_gpu_data();
  Dtype* masks128 = top[7]->mutable_gpu_data();

  FWD<Dtype> <<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), bottom[0]->gpu_data(), channels_, height_, width_, masks8, masks16, masks24, masks32, masks48, masks64, masks96, masks128, len8_, len16_, len24_, len32_, len48_, len64_, len96_, len128_);
}

template <typename Dtype>
__global__ void BACK(const int nthreads,
   const int channels, const int height, const int width, const Dtype* diff_8,const Dtype* diff_16,const Dtype* diff_24,const Dtype* diff_32,const Dtype* diff_48,const Dtype* diff_64,const Dtype* diff_96,const Dtype* diff_128, const int len8, const int len16, const int len24, const int len32, const int len48, const int len64, const int len96, const int len128, Dtype* bottom_diff
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / (channels*height*width); 
    if (n < len8)
        bottom_diff[index]=diff_8[index];
    else{
    if (n < len16+len8)
        bottom_diff[index]=diff_16[index-len8*channels*height*width];
    else{
    if (n < len24+len16+len8)
        bottom_diff[index]=diff_24[index-len8*channels*height*width-len16*channels*height*width];
    else{
    if (n < len32+len24+len16+len8)
        bottom_diff[index]=diff_32[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width];
    else{
    if (n < len48+len32+len24+len16+len8)
        bottom_diff[index]=diff_48[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width];
    else{
    if (n < len64+len48+len32+len24+len16+len8)
        bottom_diff[index]=diff_64[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width-len48*channels*height*width];
    else{
    if (n < len96+len64+len48+len32+len24+len16+len8)
        bottom_diff[index]=diff_96[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width-len48*channels*height*width-len64*channels*height*width];
    else{
        bottom_diff[index]=diff_128[index-len8*channels*height*width-len16*channels*height*width-len24*channels*height*width-len32*channels*height*width-len48*channels*height*width-len64*channels*height*width-len96*channels*height*width];
    }}}}}}}

  } //CUDA_KERNEL_LOOP
}

template <typename Dtype>
void SplitMasks8Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* diff_8 = top[0]->gpu_diff();
  const Dtype* diff_16 = top[1]->gpu_diff();
  const Dtype* diff_24 = top[2]->gpu_diff();
  const Dtype* diff_32 = top[3]->gpu_diff();
  const Dtype* diff_48 = top[4]->gpu_diff();
  const Dtype* diff_64 = top[5]->gpu_diff();
  const Dtype* diff_96 = top[6]->gpu_diff();
  const Dtype* diff_128 = top[7]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  BACK<Dtype> <<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->count(), channels_, height_, width_, diff_8, diff_16, diff_24, diff_32, diff_48, diff_64, diff_96, diff_128, len8_, len16_, len24_, len32_, len48_, len64_, len96_, len128_, bottom_diff);
  
}


INSTANTIATE_LAYER_GPU_FUNCS(SplitMasks8Layer);

}  // namespace caffe

