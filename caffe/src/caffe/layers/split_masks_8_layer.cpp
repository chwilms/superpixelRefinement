#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <math.h>
#include <cmath>

#include <boost/math/special_functions/round.hpp>

#include "caffe/layers/split_masks_8_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SplitMasks8Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SplitMasks8Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  len8_ = bottom[1]->num();
  len16_ = bottom[2]->num();
  len24_ = bottom[3]->num();
  len32_ = bottom[4]->num();
  len48_ = bottom[5]->num();
  len64_ = bottom[6]->num();
  len96_ = bottom[7]->num();
  len128_ = bottom[8]->num();
  if (bottom[0]->num() > 1 and not(bottom[0]->num() == 1000 and (len8_+len16_+len24_+len32_+len48_+len64_+len96_+len128_)==8))
    CHECK_EQ(bottom[0]->num(), len8_+len16_+len24_+len32_+len48_+len64_+len96_+len128_) << "Output must have same number of masks as input";

  top[0]->Reshape(len8_, channels_, height_, width_);
  top[1]->Reshape(len16_, channels_, height_, width_);
  top[2]->Reshape(len24_, channels_, height_, width_);
  top[3]->Reshape(len32_, channels_, height_, width_);
  top[4]->Reshape(len48_, channels_, height_, width_);
  top[5]->Reshape(len64_, channels_, height_, width_);
  top[6]->Reshape(len96_, channels_, height_, width_);
  top[7]->Reshape(len128_, channels_, height_, width_);
}

template <typename Dtype>
void SplitMasks8Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SplitMasks8Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(SplitMasks8Layer);
#endif

INSTANTIATE_CLASS(SplitMasks8Layer);
REGISTER_LAYER_CLASS(SplitMasks8);


}  // namespace caffe
