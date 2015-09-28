#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void AffineForward(const int nthreads, const Dtype* bottom_data,
		const int num, const int channels, const int height,
		const int width, const Dtype* params, const int num_parameters,
		Dtype* top_data, const int grid_h, const int grid_w) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype pwB, phB, tl, bl, tr, br;
			int ipwB, iphB;
			int pw = index % grid_w;
			int ph = (index / grid_w) % grid_h;
			int c = (index / grid_w / grid_h) % channels;
			int n = index / grid_w / grid_h / channels;

			bottom_data += (n * channels + c) * height * width;
			params += n*num_parameters;
			phB = params[0] * ph + params[1] * pw + params[2];
			pwB = params[3] * ph + params[4] * pw + params[5];
			// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
			phB = max(0., min(static_cast<Dtype>(height - 1), phB));
			pwB = max(0., min(static_cast<Dtype>(width - 1), pwB));

			ipwB = floor(pwB);
			iphB = floor(phB);
			tl = bottom_data[iphB*width + ipwB];
			bl = bottom_data[(iphB + 1)*width + ipwB];
			tr = bottom_data[iphB*width + (ipwB + 1)];
			br = bottom_data[(iphB + 1)*width + (ipwB + 1)];
			top_data[index] =	tl*(1 - (phB - iphB)) * (1 - (pwB - ipwB)) +
								bl*((phB - iphB)) * (1 - (pwB - ipwB)) +
								tr*(1 - (phB - iphB)) * ((pwB - ipwB)) +
								br*((phB - iphB)) * ((pwB - ipwB));
		}
	}
	
	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();
		const Dtype* params = bottom[0]->gpu_data();
		for (int i = 1; i < bottom.size(); ++i) {
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - 1];
			const Dtype* bottom_data = bottom_->gpu_data();
			Dtype* top_data = top_->mutable_gpu_data();

			int count = top_->count();

			switch (spatial_transformer_param.type()) {
			case SpatialTransformerParameter_TransformType_AFFINE:
				// NOLINT_NEXT_LINE(whitespace/operators)
				AffineForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					count, bottom_data, bottom_->num(), bottom_->channels(),
					bottom_->height(), bottom_->width(), params, num_parameters_, top_data, grid_h_, grid_w_);
				break;
			default:
				LOG(FATAL) << "Unknown tranform.";
			}
			CUDA_POST_KERNEL_CHECK;
		}
	}


	template <typename Dtype>
	__global__ void AffineBackward(const int nthreads, const Dtype* top_diff,
		const int num, const int channels, const int grid_h, const int grid_w,
		const Dtype * params, Dtype* param_diff, const int num_parameters,
		const Dtype* bottom_data, Dtype* bottom_diff, const int height, const int width) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the grid index
			Dtype dhdT[6]; // temporary storage for partial derivatives with respect to transform parameters
			Dtype dwdT[6];
			int pw = index % grid_w;
			int ph = (index / grid_w) % grid_h;
			int c = (index / grid_w/ grid_h) % channels;
			int n = index / grid_w / grid_h / channels;
			Dtype pwB, phB, tl, bl, tr, br;
			int ipwB, iphB;

			top_diff += (n * channels + c) * grid_h * grid_w;
			params += num_parameters*n;
			phB = params[0] * ph + params[1] * pw + params[2];
			pwB = params[3] * ph + params[4] * pw + params[5];
			
			// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
			phB = max(0., min(static_cast<Dtype>(height - 1), phB));
			pwB = max(0., min(static_cast<Dtype>(width - 1), pwB));

			// This will be similar for other transformation (with same sampling kernel)
			ipwB = floor(pwB);
			iphB = floor(phB);
			bottom_diff[iphB*width + ipwB] +=
				top_diff[index] * (1 - (phB - iphB)) * (1 - (pwB - ipwB));
			bottom_diff[(iphB + 1)*width + ipwB] +=
				top_diff[index] * ((phB - iphB)) * (1 - (pwB - ipwB));
			bottom_diff[iphB*width + (ipwB + 1)] +=
				top_diff[index] * ((phB - iphB)) * (1 - (pwB - ipwB));
			bottom_diff[(iphB + 1)*width + (ipwB + 1)] +=
				top_diff[index] * ((phB - iphB)) * ((pwB - ipwB));

			tl = bottom_data[iphB*width + ipwB];
			bl = bottom_data[(iphB + 1)*width + ipwB];
			tr = bottom_data[iphB*width + (ipwB + 1)];
			br = bottom_data[(iphB + 1)*width + (ipwB + 1)];

			// This depends on the transformation function:
			dhdT[0] = ph; dhdT[1] = pw; dhdT[2] = 1;
			dhdT[3] = 0;  dhdT[4] = 0;  dhdT[5] = 0;
			dwdT[0] = 0;  dwdT[1] = 0;  dwdT[2] = 0;
			dwdT[3] = ph; dwdT[4] = pw; dwdT[5] = 1;

			// This will be similar for other transformations (except with all partial derivatives)
			for (int param_it = 0; param_it < num_parameters; param_it++){
				param_diff[param_it] += tl * -(1 - (pwB - ipwB)) * dhdT[param_it] +
					tl * -(1 - (phB - iphB)) * dwdT[param_it] +
					bl * -(1 - (pwB - ipwB)) * dhdT[param_it] +
					bl *  ((phB - iphB)) * dwdT[param_it] +
					tr *  ((pwB - ipwB)) * dhdT[param_it] +
					tr * -(1 - (phB - iphB)) * dwdT[param_it] +
					br *  ((pwB - ipwB)) * dhdT[param_it] +
					br *  ((phB - iphB)) * dwdT[param_it];
			}

		}
	}



	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0]) {
			return;
		}
		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();

		const Dtype* params = bottom[0]->gpu_data();
		Dtype* param_diff = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), Dtype(0.), param_diff);

		for (int i = 1; i < bottom.size(); ++i) {
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - 1];

			const Dtype* top_diff = top_->gpu_diff();
			Dtype* bottom_diff = bottom_->mutable_gpu_diff();
			const Dtype* bottom_data = bottom_->gpu_data();

			caffe_gpu_set(bottom_->count(), Dtype(0.), bottom_diff);

			// This is different from the pooling example
			// since I am iterating over the resampling grid
			// instead of over the bottom pixels
			// the original was 
			//  const int count = bottom_->count();
			const int count = top_->count();
			switch (spatial_transformer_param.type()) {
			case SpatialTransformerParameter_TransformType_AFFINE:
				// NOLINT_NEXT_LINE(whitespace/operators)
				AffineBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					count, top_diff, top_->num(), top_->channels(),
					grid_h_, grid_w_, params, param_diff, num_parameters_, bottom_data, bottom_diff,
					bottom_->height(), bottom_->width());
				break;
			default:
				LOG(FATAL) << "Unknown tranform.";
			}
			CUDA_POST_KERNEL_CHECK;
		}
	}


		INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);


}  // namespace caffe
