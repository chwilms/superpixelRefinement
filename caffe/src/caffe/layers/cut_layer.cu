#include "caffe/layers/cut_layer.hpp"
#include <iostream>


namespace caffe {

template <typename Dtype>
__global__ void CutForward(const int nthreads, const Dtype* bottom_data, const int heightSmall, const int widthSmall, const int heightBig, const int widthBig, const int channels, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int b = index / (heightSmall * widthSmall * channels);
    int n = index % (heightSmall * widthSmall * channels);
    int c = n / (heightSmall * widthSmall);
    n = n % (heightSmall * widthSmall);
    int h = n / widthSmall;
    int w = n % widthSmall;

//    top_data[b*heightBig * widthBig * channels + c*heightBig * widthBig+ h*heightBig +w] = bottom_data[index];
    top_data[index] = bottom_data[b*heightBig * widthBig * channels + c*heightBig * widthBig+ h*heightBig +w];
  }
}

template <typename Dtype>
void CutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  CutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, heightSmall_, widthSmall_, heightBig_, widthBig_, channels_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void CutBackward(const int nthreads, const Dtype* top_diff, const int heightSmall, const int widthSmall, const int heightBig, const int widthBig, const int channels, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int b = index / (heightSmall * widthSmall * channels);
    int n = index % (heightSmall * widthSmall * channels);
    int c = n / (heightSmall * widthSmall);
    n = n % (heightSmall * widthSmall);
    int h = n / widthSmall;
    int w = n % widthSmall;

    bottom_diff[b*heightBig * widthBig * channels + c*heightBig * widthBig+ h*heightBig +w] = top_diff[index];
  }
}

template <typename Dtype>
void CutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

  CutBackward<Dtype> <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_diff, heightSmall_, widthSmall_, heightBig_, widthBig_, channels_, bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS(CutLayer);


}
