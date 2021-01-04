#include "caffe/layers/cut_layer.hpp"
#include <iostream>


namespace caffe {

  template <typename Dtype>
  void CutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  }

  template <typename Dtype>
  void CutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->shape(0),bottom[0]->shape(1),bottom[0]->shape(2)-1,bottom[0]->shape(3)-1);
    num_ = bottom[0]->shape(0);
    channels_ = bottom[0]->shape(1);
    heightBig_ = bottom[0]->shape(2);
    widthBig_ = bottom[0]->shape(3);
    heightSmall_ = bottom[0]->shape(2)-1;
    widthSmall_ = bottom[0]->shape(3)-1;
  }

  template <typename Dtype>
  void CutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  }

  template <typename Dtype>
  void CutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  }

#ifdef CPU_ONLY
STUB_GPU(CutLayer);
#endif

INSTANTIATE_CLASS(CutLayer);
REGISTER_LAYER_CLASS(Cut);


}
