#include "caffe/layers/convertIndices.hpp"
#include <iostream>

namespace caffe {

  template <typename Dtype>
  void ConvertIndicesLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  }

  template <typename Dtype>
  void ConvertIndicesLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->shape(0), 1);// 
    CHECK_EQ(bottom[0]->shape(1), 1);
    CHECK_EQ(bottom[0]->shape(2), 1);
    CHECK_EQ(bottom[1]->shape(0), 1);
    CHECK_EQ(bottom[1]->shape(1), 1);
    CHECK_EQ(bottom[1]->shape(2), 1);
    bottomNum_ = bottom[0]->shape(3);
    topNum_ = bottom[1]->shape(3);
//    const Dtype* bottom_data = bottom[2]->cpu_data();
//    int c = 0;
//    for (int i = 0; i < bottom[2]->shape(3); ++i){
//      c+=bottom_data[i];
//    }
//std::cout<<"num of flags in convertindices "<<c<<" "<<bottom[2]->shape(3)<<"\n";
    top[0]->Reshape(1, 1, 1, topNum_);
    top[1]->Reshape(bottomNum_,1,1,1);
  }

  template <typename Dtype>
  void ConvertIndicesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_data_2 = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* top_data_2 = top[1]->mutable_cpu_data();
//    caffe_set(top[0]->count(), Dtype(0), top_data);
    for (int i = 0; i < topNum_; ++i){
      top_data[i]=0;
    }

    for (int i = 0; i < bottomNum_; ++i){
      int index = bottom_data[i];
      top_data[index]=1;
      top_data_2[i]=bottom_data_2[index];
    }
  }

  template <typename Dtype>
  void ConvertIndicesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//    std::cout<<"Hello from backward convertIndices\n";
//    NOT_IMPLEMENTED;
  }

#ifdef CPU_ONLY
STUB_GPU(ConvertIndicesLayer);
#endif

INSTANTIATE_CLASS(ConvertIndicesLayer);
REGISTER_LAYER_CLASS(ConvertIndices);


}
