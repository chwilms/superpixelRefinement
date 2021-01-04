#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <math.h>
#include <cmath>

#include <boost/math/special_functions/round.hpp>

#include "caffe/layers/scale_flags_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ScaleFlagsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  scalefactor_ = this->layer_param_.scale_flags_param().scalefactor();
}

template <typename Dtype>
void ScaleFlagsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  height_seg_ = bottom[1]->height();
  width_seg_ = bottom[1]->width();

  CHECK_EQ((height_+1)*(width_+1), bottom[2]->shape(3)) << "Number of flags must be equal to number of locations between pixels in feature map";
  top[0]->Reshape(1, 1, 1, (height_seg_+1)*(width_seg_+1));

}

template <typename Dtype>
void ScaleFlagsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ScaleFlagsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(ScaleFlagsLayer);
#endif

INSTANTIATE_CLASS(ScaleFlagsLayer);
REGISTER_LAYER_CLASS(ScaleFlags);


}  // namespace caffe
