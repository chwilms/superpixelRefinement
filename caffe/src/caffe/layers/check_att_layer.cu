#include "caffe/layers/check_att_layer.hpp"
#include <iostream>


namespace caffe {

template <typename Dtype>
__global__ void CheckAttForward(const int n, const Dtype* bottom_obj_data, const Dtype* bottom_noObj_data, Dtype* top_obj_data, Dtype* top_noObj_data) {
  CUDA_KERNEL_LOOP(index, n) {
    top_obj_data[index] = bottom_obj_data[index];
    top_noObj_data[index] = bottom_noObj_data[index];
  }
  top_obj_data[0] = 1.00;
  top_noObj_data[0] = 0.00;
}

template <typename Dtype>
void CheckAttLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_obj_data = bottom[0]->gpu_data();
  const Dtype* bottom_noObj_data = bottom[1]->gpu_data();
  Dtype* top_obj_data = top[0]->mutable_gpu_data();
  Dtype* top_noObj_data = top[1]->mutable_gpu_data();
  const int count = bottom[0]->count();
  CheckAttForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_obj_data, bottom_noObj_data, top_obj_data, top_noObj_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void CheckAttLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_LAYER_GPU_FUNCS(CheckAttLayer);


}
