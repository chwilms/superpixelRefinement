#include "caffe/layers/fake.hpp"
#include <iostream>
#include <algorithm> 
#include <ctime>

namespace caffe {

int diff_ms(timeval t1, timeval t2)
{
    return (((t1.tv_sec - t2.tv_sec) * 1000000) + 
            (t1.tv_usec - t2.tv_usec))/1000;
}

  template <typename Dtype>
  void FakeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  }

  template <typename Dtype>
  void FakeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
if(bottom[0]->shape(0) == 1){
      for (int i = 1; i < bottom.size(); ++i) {
        top[i-1]->Reshape(1,1, 1, 1);
        top[bottom.size()-1+i-1]->Reshape(1, 1, 1, 1);
      }
    }
    else{
      const Dtype* bottom_data = bottom[0]->cpu_data();
      int counterOverall = 0;
      for (int i = 1; i < bottom.size(); ++i) {
        const Dtype* flags = bottom[i]->cpu_data();
        int counterLocal = 0;
        int counterLocal2 = 0;
        for (int j = 0; j < bottom[i]->shape(3); ++j) {
          if( flags[j]==1){
            if(bottom_data[counterOverall] == 1){
              counterLocal2+=1;
            }
            counterLocal += 1;
            counterOverall += 1;
          }
        }
        top[i-1]->Reshape(counterLocal,1, 1, 1);
        top[bottom.size()-1+i-1]->Reshape(counterLocal2, 1, 1, 1);
      }
    }
  }

  template <typename Dtype>
  void FakeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    int counterOverall = 0;
    for (int i = 1; i < bottom.size(); ++i) {
      const Dtype* flags = bottom[i]->cpu_data();
      int counterLocal = 0;
      Dtype* top_data = top[i-1]->mutable_cpu_data();
      for (int j = 0; j < bottom[i]->shape(3); ++j) {
        if( flags[j]==1){
          top_data[counterLocal]=bottom_data[counterOverall];
          counterLocal += 1;
          counterOverall += 1;
        }
      }
    }
  }

  template <typename Dtype>
  void FakeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }

#ifdef CPU_ONLY
STUB_GPU(FakeLayer);
#endif

INSTANTIATE_CLASS(FakeLayer);
REGISTER_LAYER_CLASS(Fake);


}
