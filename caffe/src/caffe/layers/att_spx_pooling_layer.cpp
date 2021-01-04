#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <math.h>
#include <cmath>

#include <boost/math/special_functions/round.hpp>

#include "caffe/layers/att_spx_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void AttSpxPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SuperpixelPoolingParameter superpixel_pooling_param = this->layer_param_.superpixel_pooling_param();
}

template <typename Dtype>
void AttSpxPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[1]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  batchEntries_ = bottom[1]->num();
  height_ = bottom[1]->height();
  width_ = bottom[1]->width();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "Input must have only 1 "
      << "batch entry (1, 1, height, width)";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "Input must have only 1 "
      << "batch entry (1, 1, height, width)";
  CHECK_EQ(bottom[1]->channels(), 1) << "Input must have only 1 "
      << "channel entry (1, 1, height, width)";

  maxValSeg_=0;
  const Dtype* bottom_data = bottom[2]->cpu_data();
  for (int i = 0; i < bottom[2]->count(); ++i)
    maxValSeg_=max(maxValSeg_,bottom_data[i]);

  top[0]->Reshape(bottom[0]->num(),1,1, maxValSeg_+1);
  top[1]->Reshape(bottom[0]->num(),1,1,maxValSeg_+1);
}

template <typename Dtype>
void AttSpxPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void AttSpxPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(AttSpxpoolingLayer);
#endif

INSTANTIATE_CLASS(AttSpxPoolingLayer);
REGISTER_LAYER_CLASS(AttSpxPooling);


}  // namespace caffe
