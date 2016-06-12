#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/seginfogain_loss_layer.hpp"

#define kLOG_THRESHOLDb 1e-20;

namespace caffe {
template <typename Dtype>
__global__ void SegInfogainLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
		  Dtype* counts, const Dtype* infogain_mat) {
	const int channels = dim / spatial_dim;
	CUDA_KERNEL_LOOP(index, nthreads) {
	const int n = index / dim; // case #
    const int s = index % spatial_dim; // voxel #
	const int k = (index % dim) / spatial_dim; // label #
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
	}
	else {
		loss[index] = -infogain_mat[label_value * channels + k] * log(max(prob_data[n * dim + k * spatial_dim + s],
			Dtype(1e-20)));

		counts[index] = 1;
	}
  }
}

template <typename Dtype>
void SegInfogainLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* infogain_mat = NULL;
  if (bottom.size() < 3) {
	  infogain_mat = infogain_.gpu_data();
  }
  else {
	  infogain_mat = bottom[2]->gpu_data();
  }

  const Dtype* prob_datac = prob_.cpu_data();
  
  const int dim = prob_.count() / outer_num_; //voxels x labels
  const int numLabels = prob_.count() / outer_num_ / inner_num_;
  const int nthreads = outer_num_ * inner_num_ * numLabels;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SegInfogainLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
	  outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, infogain_mat);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);

  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(nthreads, counts, &count);
	loss /= count / numLabels; //count is incremented once per label for every included voxel
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SegInfogainLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
		  const int ignore_label_, Dtype* counts, const Dtype* infogain_mat, const Dtype* infogain_sum) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
	const int n = index / dim; // voxel #
    const int s = index % spatial_dim; // case #
	const int k = (index % dim) / spatial_dim; // label #
	const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
		bottom_diff[n * dim + k * spatial_dim + s] *= infogain_sum[label_value];
		bottom_diff[n * dim + k * spatial_dim + s] -= infogain_mat[label_value * channels + k];
		counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SegInfogainLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down.size() > 2 && propagate_down[2]) {
	  LOG(FATAL) << this->type()
		  << " Layer cannot backpropagate to infogain inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
	caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
	const Dtype* label = bottom[1]->gpu_data();
	const Dtype* infogain_mat = NULL;
    const int dim = prob_.count() / outer_num_;
	const int numLabels = prob_.count() / outer_num_ / inner_num_;
	const int nthreads = outer_num_ * inner_num_ * numLabels;
	if (bottom.size() < 3) {
		infogain_mat = infogain_.gpu_data();
	}
	else {
		infogain_mat = bottom[2]->gpu_data();
		const Dtype* infogain_mat_cpu = bottom[2]->cpu_data();
		Dtype* infogain_sum_cpu = infogain_sum_.mutable_cpu_data();
		for (int labelIt = 0; labelIt < numLabels; labelIt++)
		{
			infogain_sum_cpu[labelIt] = caffe_cpu_asum(numLabels, infogain_mat_cpu + labelIt * numLabels);

		}
	}
	const Dtype* infogain_sum = infogain_sum_.gpu_data();

    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
	SegInfogainLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
		outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, infogain_mat, infogain_sum);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / (count/numLabels), bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SegInfogainLossLayer);


}  // namespace caffe
