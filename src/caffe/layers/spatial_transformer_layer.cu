#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void AffineForward(const int nthreads, const Dtype* bottom_data,
		const int num, const int channels, const int height,
		const int width, const Dtype* params_, const int num_parameters,
		Dtype* top_data, const int grid_h, const int grid_w, bool inverse) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype pwB, phB, tl, bl, tr, br;
			int ipwB, iphB;
			int pw = index % grid_w;
			int ph = (index / grid_w) % grid_h;
			int c = (index / grid_w / grid_h) % channels;
			int n = index / grid_w / grid_h / channels;

			bottom_data += (n * channels + c) * height * width;
			params_ += n*num_parameters;
			if (!inverse) {
				phB = params_[0] * ph + params_[1] * pw + params_[2];
				pwB = params_[3] * ph + params_[4] * pw + params_[5];
			}
			else {
				Dtype det = params_[0] * params_[4] - params_[1] * params_[3];
				Dtype inv0 = params_[4] / det;
				Dtype inv1 = -params_[1] / det;
				Dtype inv3 = -params_[3] / det;
				Dtype inv4 = params_[0] / det;
				Dtype inv2 = -(inv0 * params_[2] + inv1 * params_[5]);
				Dtype inv5 = -(inv3 * params_[2] + inv4 * params_[5]);

				phB = inv0 * ph + inv1 * pw + inv2;
				pwB = inv3 * ph + inv4 * pw + inv5;

			}

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
		const Dtype* params;
		int firstDataBlob;
		if (spatial_transformer_param.const_params().size())
		{
			params = constParamsBlob_.gpu_data();
			firstDataBlob = 0;
		}
		else
		{

			params = bottom[0]->gpu_data();
			firstDataBlob = 1;

		}
		for (int i = firstDataBlob; i < bottom.size(); ++i) {
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - firstDataBlob];
			const Dtype* bottom_data = bottom_->gpu_data();
			Dtype* top_data = top_->mutable_gpu_data(); 
			
			
			
			//
			// This line causes an error on the second top if there are constant parameters
			//



			int count = top_->count();

			switch (spatial_transformer_param.type()) {
			case SpatialTransformerParameter_TransformType_AFFINE:
				// NOLINT_NEXT_LINE(whitespace/operators)
				AffineForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					count, bottom_data, bottom_->num(), bottom_->channels(),
					bottom_->height(), bottom_->width(), params, num_parameters_, top_data, grid_h_, grid_w_, false);
				break;
			case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
				// NOLINT_NEXT_LINE(whitespace/operators)
				AffineForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					count, bottom_data, bottom_->num(), bottom_->channels(),
					bottom_->height(), bottom_->width(), params, num_parameters_, top_data, grid_h_, grid_w_, true);
				break;
			default:
				LOG(FATAL) << "Unknown tranform: " << SpatialTransformerParameter_TransformType_INVERSE_AFFINE;
			}
			CUDA_POST_KERNEL_CHECK;

		}
	}


	template <typename Dtype>
	__global__ void AffineBackward(const int nthreads, const Dtype* top_diff,
		const int num, const int channels, const int grid_h, const int grid_w,
		const Dtype * params_, Dtype* param_diff, const int num_parameters,
		const Dtype* bottom_data, Dtype* bottom_diff, const int height, const int width, bool inverse, bool propagate_down_data, bool propagate_down_param) {

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
			bottom_diff += (n * channels + c) * height * width;
			bottom_data += (n * channels + c) * height * width;
			params_ += num_parameters*n;
			param_diff += num_parameters*n;
			if (!inverse) {
				phB = params_[0] * ph + params_[1] * pw + params_[2];
				pwB = params_[3] * ph + params_[4] * pw + params_[5];
			}
			else{
				Dtype det = params_[0] * params_[4] - params_[1] * params_[3];
				Dtype inv0 = params_[4] / det;
				Dtype inv1 = -params_[1] / det;
				Dtype inv3 = -params_[3] / det;
				Dtype inv4 = params_[0] / det;
				Dtype inv2 = -(inv0 * params_[2] + inv1 * params_[5]);
				Dtype inv5 = -(inv3 * params_[2] + inv4 * params_[5]);

				phB = inv0 * ph + inv1 * pw + inv2;
				pwB = inv3 * ph + inv4 * pw + inv5;

			}
			
			// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
			phB = max(0., min(static_cast<Dtype>(height - 1), phB));
			pwB = max(0., min(static_cast<Dtype>(width - 1), pwB));

			// This will be similar for other transformation (with same sampling kernel)
			ipwB = floor(pwB);
			iphB = floor(phB);
			if (propagate_down_data)
			{
				caffe_gpu_atomic_add(top_diff[ph*grid_w+pw] * (1 - (phB - iphB)) * (1 - (pwB - ipwB)), bottom_diff + iphB*width + ipwB);
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * ((phB - iphB)) * (1 - (pwB - ipwB)), bottom_diff + (iphB + 1)*width + ipwB);
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * ((phB - iphB)) * (1 - (pwB - ipwB)), bottom_diff + iphB*width + (ipwB + 1));
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * ((phB - iphB)) * ((pwB - ipwB)), bottom_diff + (iphB + 1)*width + (ipwB + 1));
			}
			if (propagate_down_param)
			{
				tl = bottom_data[iphB*width + ipwB];
				bl = bottom_data[(iphB + 1)*width + ipwB];
				tr = bottom_data[iphB*width + (ipwB + 1)];
				br = bottom_data[(iphB + 1)*width + (ipwB + 1)];

				// This depends on the transformation function:
				if (!inverse) {

					dhdT[0] = ph; dhdT[1] = pw; dhdT[2] = 1;
					dhdT[3] = 0;  dhdT[4] = 0;  dhdT[5] = 0;
					dwdT[0] = 0;  dwdT[1] = 0;  dwdT[2] = 0;
					dwdT[3] = ph; dwdT[4] = pw; dwdT[5] = 1;
				}
				else {
					const Dtype *p = params_;
					Dtype det = (p[0] * p[4] - p[1] * p[3]);
					Dtype det2 = det*det;
					// From Matlab sym package: [simplify(diff(inv([p0 p1; p3 p4])*[ph - p2; pw - p5], p0)*(p0*p4 - p1*p3) ^ 2); simplify(diff(inv([p0 p1; p3 p4])*[ph - p2; pw - p5], p1)*(p0*p4 - p1*p3) ^ 2); simplify(diff(inv([p0 p1; p3 p4])*[ph - p2; pw - p5], p2)*(p0*p4 - p1*p3) ^ 2); simplify(diff(inv([p0 p1; p3 p4])*[ph - p2; pw - p5], p3)*(p0*p4 - p1*p3) ^ 2); simplify(diff(inv([p0 p1; p3 p4])*[ph - p2; pw - p5], p4)*(p0*p4 - p1*p3) ^ 2); simplify(diff(inv([p0 p1; p3 p4])*[ph - p2; pw - p5], p5)*(p0*p4 - p1*p3) ^ 2)]
					dhdT[0] = (-p[4] * (p[1] * p[5] - p[2] * p[4] + p[4] * ph - p[1] * pw)) / det2;
					dwdT[0] = (p[3] * (p[1] * p[5] - p[2] * p[4] + p[4] * ph - p[1] * pw)) / det2;
					dhdT[1] = (p[4] * (p[0] * p[5] - p[2] * p[3] + p[3] * ph - p[0] * pw)) / det2;
					dwdT[1] = (-p[3] * (p[0] * p[5] - p[2] * p[3] + p[3] * ph - p[0] * pw)) / det2;
					dhdT[2] = (p[1] * p[3] * p[4] - p[0] * p[4] * p[0] * p[4]) / det2;
					dwdT[2] = (p[3] * (p[0] * p[4] - p[1] * p[3])) / det2;
					dhdT[3] = (p[1] * (p[1] * p[5] - p[2] * p[4] + p[4] * ph - p[1] * pw)) / det2;
					dwdT[3] = (-p[0] * (p[1] * p[5] - p[2] * p[4] + p[4] * ph - p[1] * pw)) / det2;
					dhdT[4] = (-p[1] * (p[0] * p[5] - p[2] * p[3] + p[3] * ph - p[0] * pw)) / det2;
					dwdT[4] = (p[0] * (p[0] * p[5] - p[2] * p[3] + p[3] * ph - p[0] * pw)) / det2;
					dhdT[5] = (p[1] * (p[0] * p[4] - p[1] * p[3])) / det2;
					dwdT[5] = (-p[4] * p[0] * p[4] * p[0] + p[1] * p[3] * p[0]) / det2;

				}
				// This will be similar for other transformations (except with all partial derivatives)
				for (int param_it = 0; param_it < num_parameters; param_it++){
					caffe_gpu_atomic_add(
						tl * -(1 - (pwB - ipwB)) * dhdT[param_it] +
						tl * -(1 - (phB - iphB)) * dwdT[param_it] +
						bl * -(1 - (pwB - ipwB)) * dhdT[param_it] +
						bl *  ((phB - iphB)) * dwdT[param_it] +
						tr *  ((pwB - ipwB)) * dhdT[param_it] +
						tr * -(1 - (phB - iphB)) * dwdT[param_it] +
						br *  ((pwB - ipwB)) * dhdT[param_it] +
						br *  ((phB - iphB)) * dwdT[param_it], param_diff+param_it);
				}
			}

		}
	}



	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();

		const Dtype* params;
		int firstDataBlob;
		Dtype* param_diff;
		bool hasConstParams;
		if (spatial_transformer_param.const_params().size())
		{
			params = constParamsBlob_.gpu_data();
			firstDataBlob = 0;
			param_diff = constParamsBlob_.mutable_gpu_diff();
			caffe_gpu_set(constParamsBlob_.count(), Dtype(0.), param_diff);
			hasConstParams = true;
		}
		else
		{

			params = bottom[0]->gpu_data();
			firstDataBlob = 1;
			param_diff = bottom[0]->mutable_gpu_diff();
			caffe_gpu_set(bottom[0]->count(), Dtype(0.), param_diff);
			hasConstParams = false;
		}

		for (int i = firstDataBlob; i < bottom.size(); ++i) {
			if ((hasConstParams && !propagate_down[i]) || (!hasConstParams && !propagate_down[i] && !propagate_down[0]))
			{
				continue;
			}
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - firstDataBlob];

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
					bottom_->height(), bottom_->width(), false, propagate_down[i], propagate_down[0] && !hasConstParams);
				break;
			case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
				// NOLINT_NEXT_LINE(whitespace/operators)
				AffineBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					count, top_diff, top_->num(), top_->channels(),
					grid_h_, grid_w_, params, param_diff, num_parameters_, bottom_data, bottom_diff,
					bottom_->height(), bottom_->width(), true, propagate_down[i], propagate_down[0] && !hasConstParams);
				break;
			default:
				LOG(FATAL) << "Unknown tranform: " << SpatialTransformerParameter_TransformType_INVERSE_AFFINE;
			}
			CUDA_POST_KERNEL_CHECK;
		}
	}


		INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);


}  // namespace caffe
