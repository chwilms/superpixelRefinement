#include <vector>

#include "caffe/layers/normalized_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}


template <typename Dtype>
void NormalizedSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) { LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs."; }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int width = bottom[0]->shape(2); const int height = bottom[0]->shape(3);
    const int factor = bottom[0]->shape(1) * bottom[0]->shape(2) * bottom[0]->shape(3);

    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, ignore_label_, target, bottom_diff);
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / ( num * width * height ), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(NormalizedSigmoidCrossEntropyLossLayer);


}  // namespace caffe
