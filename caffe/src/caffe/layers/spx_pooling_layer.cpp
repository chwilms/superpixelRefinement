#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <math.h>
#include <cmath>

#include <boost/math/special_functions/round.hpp>

#include "caffe/layers/spx_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void SpxPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SuperpixelPoolingParameter superpixel_pooling_param = this->layer_param_.superpixel_pooling_param();
}

template <typename Dtype>
void SpxPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  std::cout<<"SpxPoolingLayer reshape\n";
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_feat_ = bottom[0]->height();
  width_feat_ = bottom[0]->width();
  height_spx_ = bottom[1]->height();
  width_spx_ = bottom[1]->width();
  scalefactor_ = ((float) height_feat_)/((float) height_spx_);
  CHECK_EQ(bottom[1]->num(), 1) << "Input must have only 1 "
      << "batch entry (1, 1, height, width)";
  CHECK_EQ(bottom[1]->num(), 1) << "Input must have only 1 "
      << "channel entry (1, 1, height, width)";
  CHECK_EQ(bottom[0]->num(), 1) << "Input must have only 1 "
      << "batch entry (1, channels, height, width)";
//  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2)) << "Input "
//      << "shape of input and spx map must be equivalent";
//  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3)) << "Input "
//      << "shape of input and spx map must be equivalent";

//  std::cout<<"SpxPoolingLayer reshape1\n";
  maxValSeg_=0;
  const Dtype* bottom_data = bottom[1]->cpu_data();
  for (int i = 0; i < bottom[1]->count(); ++i)
    maxValSeg_=max(maxValSeg_,bottom_data[i]);
//  std::cout<<"SpxPoolingLayer reshape2\n";
  bottom[0]->num();
  top[0]->Reshape(1, 1, 203, 256);
  top[1]->Reshape(1,1,1,203);
//  std::cout<<"SpxPoolingLayer reshape21 "<<bottom[0]->num()<<" "<<maxValSeg_+1<<" "<<channels_<<"\n";
//  std::cout<<"maxValSeg_ "<<maxValSeg_<<"\n";

  top[0]->Reshape(bottom[0]->num(), 1, maxValSeg_+1, channels_);
//  std::cout<<"SpxPoolingLayer reshape3\n";
  top[1]->Reshape(1,1,1,maxValSeg_+1);
//  std::cout<<"SpxPoolingLayer reshape done\n";
}

template <typename Dtype>
void SpxPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* spxMap = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.superpixel_pooling_param().pool()) {
  case SuperpixelPoolingParameter_PoolMethod_MAX:
    NOT_IMPLEMENTED;
    break;
  case SuperpixelPoolingParameter_PoolMethod_AVE:
    {
//      std::cout<<"Hello1 fwd"<<" "<<scalefactor_<<"\n";
      for (int i = 0; i < top_count; ++i) {
        top_data[i] = 0;
      }
      // The main loop
      std::vector<int> pxCount(maxValSeg_+1, 0);
      int numPx = height_feat_ * width_feat_;
      for (int h_s = 0; h_s < height_spx_; ++h_s) {
        float h_s_in_f = h_s * scalefactor_; //coord of spx map in feature map coords
        float hw = min(height_feat_-1.0f,max(boost::math::round(h_s_in_f),0.0f)) * width_feat_;
        for (int w_s = 0; w_s < width_spx_; ++w_s) {
          int spxId = (int)spxMap[h_s * width_spx_ + w_s];
          int spxIdTimesChannels = spxId * channels_;
//          Dtype newSample;
          float w_s_in_f = w_s * scalefactor_;
          float index = hw + min(width_feat_-1.0f,max(boost::math::round(w_s_in_f),0.0f));
/*
          int index1 = 0;
          int index2 = 0;
          int index3 = 0;
          int index4 = 0;
          float weight1 = 0.0f;
          float weight2 = 0.0f;
          float weight_w1 = 0.0f;
          float weight_w2 = 0.0f;
          float weight_h1 = 0.0f;
          float weight_h2 = 0.0f;
          if (ceilf(h_s_in_f) == h_s_in_f && ceilf(w_s_in_f) == w_s_in_f){ //no interpolation necessary
        	index1 = (int) (h_s_in_f * width_feat_ + w_s_in_f);
          }
	  if (ceilf(h_s_in_f) == h_s_in_f && ceilf(w_s_in_f) != w_s_in_f){ //interpolation only in width
                int w1_s_in_f = max(0.0f,floorf(w_s_in_f));
                weight1 = w_s_in_f - floorf(w_s_in_f);
                int w2_s_in_f = min(ceilf(w_s_in_f), width_feat_-1.0f);
                weight2 = 1-weight1;
                index1 = (int) (h_s_in_f * width_feat_ + w1_s_in_f);
                index2 = (int) (h_s_in_f * width_feat_ + w2_s_in_f);
          }
          if (ceilf(h_s_in_f) != h_s_in_f && ceilf(w_s_in_f) == w_s_in_f){ //interpolation only in height
                int h1_s_in_f = max(0.0f,floorf(h_s_in_f));
                weight1 = h_s_in_f - floorf(h_s_in_f);
                int h2_s_in_f = min(ceilf(h_s_in_f), height_feat_-1.0f);
                weight2 = 1-weight1;
                index1 = (int) (h1_s_in_f * width_feat_ + w_s_in_f);
                index2 = (int) (h2_s_in_f * width_feat_ + w_s_in_f);
          }
          if (ceilf(h_s_in_f) != h_s_in_f && ceilf(w_s_in_f) != w_s_in_f){ //interpolation in both dimensions
                int w1_s_in_f = max(0.0f,floorf(w_s_in_f));
                weight_w1 = w_s_in_f - floorf(w_s_in_f);
                int w2_s_in_f = min(ceilf(w_s_in_f), width_feat_-1.0f);
                weight_w2 = 1-weight_w1;
		int h1_s_in_f = max(0.0f,floorf(h_s_in_f));
                weight_h1 = h_s_in_f - floorf(h_s_in_f);
                int h2_s_in_f = min(ceilf(h_s_in_f), height_feat_-1.0f);
                weight_h2 = 1-weight_h1;
                index1 = (int) (h1_s_in_f * width_feat_ + w1_s_in_f);
                index2 = (int) (h1_s_in_f * width_feat_ + w2_s_in_f);
                index3 = (int) (h2_s_in_f * width_feat_ + w1_s_in_f);
                index4 = (int) (h2_s_in_f * width_feat_ + w2_s_in_f);
          }            
*/
          pxCount[spxId]+=1;
          for (int c = 0; c < channels_; ++c) {
/*
            if (ceilf(h_s_in_f) == h_s_in_f && ceilf(w_s_in_f) == w_s_in_f){ //no interpolation necessary
        	newSample = bottom_data[c * numPx + index1];
            }
            if (ceilf(h_s_in_f) == h_s_in_f && ceilf(w_s_in_f) != w_s_in_f){ //interpolation only in width
                Dtype s1 = bottom_data[c * numPx + index1];
                Dtype s2 = bottom_data[c * numPx + index2];
                newSample = s1*weight1+s2*weight2;
            }
            if (ceilf(h_s_in_f) != h_s_in_f && ceilf(w_s_in_f) == w_s_in_f){ //interpolation only in height
                Dtype s1 = bottom_data[c * numPx + index1];
                Dtype s2 = bottom_data[c * numPx + index2];
                newSample = s1*weight1+s2*weight2;
            }
            if (ceilf(h_s_in_f) != h_s_in_f && ceilf(w_s_in_f) != w_s_in_f){ //interpolation in both dimensions
                Dtype s1 = bottom_data[c * numPx + index1];
                Dtype s2 = bottom_data[c * numPx + index2];
                Dtype newH1 = s1*weight_w1+s2*weight_w2;
                s1 = bottom_data[c * numPx + index3];
                s2 = bottom_data[c * numPx + index4];
                Dtype newH2 = s1*weight_w1+s2*weight_w2;
                newSample = newH1*weight_h1+newH2*weight_h2;
            }          
*/  
            top_data[spxIdTimesChannels + c] += bottom_data[(int) (c * numPx + index)];
          }
        }
      }
      int channelOffset=0;
      for (int spxId = 0; spxId < maxValSeg_+1; ++spxId) {
        if (pxCount[spxId]>0) {
          for (int c = 0; c < channels_; ++c) {
            top_data[channelOffset + c]/=pxCount[spxId];
          }
        }
        channelOffset = channelOffset + channels_;
      }

      pxCount_ = pxCount;
//      std::cout<<"Hello12 fwd"<<"\n";
      break;
  }
  case SuperpixelPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
//      std::cout<<"Hello13 fwd"<<"\n";
}

template <typename Dtype>
void SpxPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//std::cout<<"Hello"<<"\n";
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
//  const int bottom_diff_count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* spxMap = bottom[1]->cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
    int numPx = height_feat_ * width_feat_;
    float weight = this->layer_param_.superpixel_pooling_param().weight();
    int hw = 0;
  switch (this->layer_param_.superpixel_pooling_param().pool()) {
  case SuperpixelPoolingParameter_PoolMethod_MAX:
    NOT_IMPLEMENTED;
    break;
  case SuperpixelPoolingParameter_PoolMethod_AVE:
//    for (int i = 0; i < bottom_diff_count; ++i) {
//      bottom_diff[i] = 0;
//    }
    // The main loop
//std::cout<<"Hello1"<<"\n";
    for (int h = 0; h < height_feat_; ++h) {
      for (int w = 0; w < width_feat_; ++w) {
        int spxId = (int)spxMap[(int) (h/scalefactor_ * width_spx_ + w/scalefactor_)];
        int spxIdTimesChannels = spxId * channels_;
        int index = hw + w;
        int pxC = pxCount_[spxId];
        float newWeight = weight / pxC;
        for (int c = 0; c < channels_; ++c) {
          //expects continous labeling starting form 0
          //nicht ganz korrekt
//std::cout<<"Hello2"<<" "<<h<<" "<<w<<" "<<width_spx_<<" "<<scalefactor_<<" "<<h/scalefactor_ * width_spx_ + w/scalefactor_<<" "<<height_feat_<<" "<<height_spx_<<"\n";
//std::cout<<"Hello2"<<" "<<c * height_feat_ * width_feat_ + h * width_feat_ + w<<" "<<spxId * channels_ + c<<" "<<spxId<<"\n";
        bottom_diff[c * numPx + index] =
            newWeight * top_diff[spxIdTimesChannels + c];
//std::cout<<"Hello3"<<"\n";
        }
      }
      hw += width_feat_;
    }
//std::cout<<"Hello4"<<"\n";
    break;
  case SuperpixelPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(SpxpoolingLayer);
#endif

INSTANTIATE_CLASS(SpxPoolingLayer);
REGISTER_LAYER_CLASS(SpxPooling);


}  // namespace caffe
