#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/spatial_transformer_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Configure the kernel size, padding, stride, and inputs.
		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();
		CHECK(!spatial_transformer_param.has_grid_size() !=
			!(spatial_transformer_param.has_grid_h() && spatial_transformer_param.has_grid_w()))
			<< "Resampling grid size is defined by grid_size OR grid_h and grid_w; not both";
		CHECK(spatial_transformer_param.has_grid_size() ||
			(spatial_transformer_param.has_grid_h() && spatial_transformer_param.has_grid_w()))
			<< "For non-square resampling grids both grid_h and grid_w are required.";
		if (spatial_transformer_param.has_grid_size()) {
			grid_h_ = grid_w_ = spatial_transformer_param.grid_size();
		}
		else {
			grid_h_ = spatial_transformer_param.grid_h();
			grid_w_ = spatial_transformer_param.grid_w();
		}
		CHECK_GT(grid_h_, 0) << "Resampling grid dimensions cannot be zero.";
		CHECK_GT(grid_w_, 0) << "Resampling grid dimensions cannot be zero.";

		// can have constant parameters defined, or the first bottom is the parameters

		if (spatial_transformer_param.const_params().size())
		{
			vector<int> shape(4);
			shape[3] = 1;
			shape[2] = 1;
			shape[1] = spatial_transformer_param.const_params().size();
			shape[0] = bottom[0]->shape(0);
			constParamsBlob_.Reshape(shape);
			Dtype* data = constParamsBlob_.mutable_cpu_data();
			for (int it1 = 0; it1 < shape[0]; it1++)
			{
				for (int it2 = 0; it2 < shape[1]; it2++)
				{
					data[it1*shape[1] + it2] = spatial_transformer_param.const_params().data()[it2];
				}
			}
			CHECK_EQ(top.size(), bottom.size()) << "When constant parameters are defined, must have the same number of bottoms and tops.";
		}
		else
		{
			CHECK_GT(bottom.size(), 0) << "Must have constant parameters or must have at least one input (parameters)";
			CHECK_EQ(top.size() + 1, bottom.size()) << "Must have one more bottom than top.";
		}

		// Handle the parameters: 

		// TODO genericize; For now assume affine 2d transform 6 parameters
		switch (spatial_transformer_param.type())
		{
		case SpatialTransformerParameter_TransformType_AFFINE:
		case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
			num_parameters_ = 6;
			if (spatial_transformer_param.const_params().size())
			{
				CHECK_EQ(spatial_transformer_param.const_params().size(), num_parameters_) << "Number of constant parameters must match number of parameters in transform type.";
			}
			else
			{
				CHECK_EQ(bottom[0]->channels(), num_parameters_) << "Number of channels in first input must match number of parameters in transform type.";
			}
			break;
		default:
			LOG(FATAL) << "Unknown tranform: " << SpatialTransformerParameter_TransformType_INVERSE_AFFINE;
		}


		// Propagate gradients to the parameters (as directed by backward pass).
		this->param_propagate_down_.resize(this->blobs_.size(), true);

	}

	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
			CHECK_EQ(bottom[1]->num(), bottom[bottom_id]->num()) << "Inputs must have same num.";
			CHECK_EQ(bottom[1]->height(), bottom[bottom_id]->height()) << "Inputs must have same height.";
			CHECK_EQ(bottom[1]->width(), bottom[bottom_id]->width()) << "Inputs must have same width.";
			top[bottom_id-1]->Reshape(bottom[bottom_id]->num(), bottom[bottom_id]->channels(), grid_h_, grid_w_);

		}
	}

	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();
		const Dtype* params;
		int firstDataBlob;
		if (spatial_transformer_param.const_params().size())
		{
			params = constParamsBlob_.cpu_data();
			firstDataBlob = 0;
		}
		else
		{

			params = bottom[0]->cpu_data();
			firstDataBlob = 1;
		}

		Dtype pwB, phB,tl,bl,tr,br;
		int ipwB, iphB;
		for (int i = firstDataBlob; i < bottom.size(); ++i) {
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - firstDataBlob];
			const Dtype* bottom_data = bottom_->cpu_data();
			Dtype* top_data = top_->mutable_cpu_data();
			const int width = bottom_->width();
			const int height = bottom_->height();
			caffe_set(top_->count(), Dtype(0), top_data);
			// The main loop
			for (int n = 0; n < bottom_->num(); ++n) {
				const Dtype* params_ = params + n*num_parameters_;
				for(int c = 0; c < bottom_->channels(); ++c) {
					for (int ph = 0; ph < grid_h_; ++ph) {
						for (int pw = 0; pw < grid_w_; ++pw) {
							const int index = ph * grid_w_ + pw;
							// transform point will be [phB,pwB]=transformPoint(ph,pw)
							// TODO move the pixel loop outside of the rest to avoid repeatedly calculating the weights
							switch (spatial_transformer_param.type())
							{
							case SpatialTransformerParameter_TransformType_AFFINE:
							case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
								
								if (spatial_transformer_param.type() == SpatialTransformerParameter_TransformType_AFFINE)
								{
									phB = params_[0] * ph + params_[1] * pw + params_[2];
									pwB = params_[3] * ph + params_[4] * pw + params_[5];
								}
								else if (spatial_transformer_param.type() == SpatialTransformerParameter_TransformType_INVERSE_AFFINE)
								{
									Dtype det = params_[0] * params_[4] - params_[1] * params_[3];
									Dtype inv0 = params_[4] / det;
									Dtype inv1 = -params_[1]/det;
									Dtype inv3 = -params_[3] / det;
									Dtype inv4 = params_[0] / det;
									Dtype inv2 = -(inv0 * params_[2] + inv1 * params_[5]);
									Dtype inv5 = -(inv3 * params_[2] + inv4 * params_[5]);

									phB = inv0 * ph + inv1 * pw + inv2;
									pwB = inv3 * ph + inv4 * pw + inv5;
									// Need to invert here	X = [M^{-1}Y-M^{-1}b]

								}
								// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
								phB = std::max((Dtype)0., std::min(static_cast<Dtype>(height - 1), phB));
								pwB = std::max((Dtype)0., std::min(static_cast<Dtype>(width - 1), pwB));

								ipwB = floor(pwB);
								iphB = floor(phB);
								tl = bottom_data[iphB*width + ipwB];
								bl = bottom_data[(iphB + 1)*width + ipwB];
								tr = bottom_data[iphB*width + (ipwB + 1)];
								br = bottom_data[(iphB + 1)*width + (ipwB + 1)];
								top_data[index] = tl*(1 - (phB - iphB)) * (1 - (pwB - ipwB)) +
													bl*(    (phB - iphB)) * (1 - (pwB - ipwB)) +
													tr*(1 - (phB - iphB)) * ((pwB - ipwB)) +
													br*((phB - iphB)) * ((pwB - ipwB));
								
								break;


							default:
								LOG(FATAL) << "Unknown tranform: " << SpatialTransformerParameter_TransformType_INVERSE_AFFINE;
							}
						}
					}
					bottom_data += bottom_->offset(0, 1);

					top_data += top_->offset(0, 1);
				}
							

						
			}
					
		}


	}


	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();
		vector<Dtype> dhdT(num_parameters_); // temporary storage for partial derivatives with respect to transform parameters
		vector<Dtype> dwdT(num_parameters_);
		const Dtype* params;
		int firstDataBlob;
		Dtype* bottom_param_diff;
		if (spatial_transformer_param.const_params().size())
		{
			params = constParamsBlob_.cpu_data();
			firstDataBlob = 0;
			bottom_param_diff = constParamsBlob_.mutable_cpu_diff();
		}
		else
		{

			params = bottom[0]->cpu_data();
			firstDataBlob = 1;
			bottom_param_diff = bottom[0]->mutable_cpu_diff();

		}


		Dtype pwB, phB, tl, bl, tr, br;
		int ipwB, iphB;
		for (int i = firstDataBlob; i < bottom.size(); ++i) {
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - firstDataBlob];

			const Dtype* top_diff = top_->cpu_diff();
			Dtype* bottom_diff = bottom_->mutable_cpu_diff();
			const Dtype* bottom_data = bottom_->mutable_cpu_data();
			const int width = bottom_->width();
			const int height = bottom_->height();

			caffe_set(bottom_->count(), Dtype(0), bottom_diff);
			for (int n = 0; n < bottom_->num(); ++n) {
				const Dtype* params_ = params + n*num_parameters_;
				for (int c = 0; c < bottom_->channels(); ++c) {
					for (int ph = 0; ph < grid_h_; ++ph) {
						for (int pw = 0; pw < grid_w_; ++pw) {
							const int index = ph * grid_w_ + pw;
							// transform point will be [phB,pwB]=transformPoint(ph,pw)
							// TODO move the pixel loop outside of the rest to avoid repeatedly calculating the weights
							switch (spatial_transformer_param.type())
							{
							case SpatialTransformerParameter_TransformType_AFFINE:
							case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
								if (spatial_transformer_param.type() == SpatialTransformerParameter_TransformType_AFFINE)
								{
									phB = params_[0] * ph + params_[1] * pw + params_[2];
									pwB = params_[3] * ph + params_[4] * pw + params_[5];
								}
								else if (spatial_transformer_param.type() == SpatialTransformerParameter_TransformType_INVERSE_AFFINE)
								{
									Dtype det = params_[0] * params_[4] - params_[1] * params_[3];
									Dtype inv0 = params_[4] / det;
									Dtype inv1 = -params_[1] / det;
									Dtype inv3 = -params_[3] / det;
									Dtype inv4 = params_[0] / det;
									Dtype inv2 = -(inv0 * params_[2] + inv1 * params_[5]);
									Dtype inv5 = -(inv3 * params_[2] + inv4 * params_[5]);

									phB = inv0 * ph + inv1 * pw + inv2;
									pwB = inv3 * ph + inv4 * pw + inv5;
									// Need to invert here	X = [M^{-1}Y-M^{-1}b]

								}

								// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
								phB = std::max((Dtype)0., std::min(static_cast<Dtype>(height - 1), phB));
								pwB = std::max((Dtype)0., std::min(static_cast<Dtype>(width - 1), pwB));
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
								if (!spatial_transformer_param.const_params().size())
								{

									tl = bottom_data[iphB*width + ipwB];
									bl = bottom_data[(iphB + 1)*width + ipwB];
									tr = bottom_data[iphB*width + (ipwB + 1)];
									br = bottom_data[(iphB + 1)*width + (ipwB + 1)];

		
									// This depends on the transformation function:
									if (spatial_transformer_param.type() == SpatialTransformerParameter_TransformType_AFFINE)
									{
										dhdT[0] = ph; dhdT[1] = pw; dhdT[2] = 1;
										dhdT[3] = 0;  dhdT[4] = 0;  dhdT[5] = 0;
										dwdT[0] = 0;  dwdT[1] = 0;  dwdT[2] = 0;
										dwdT[3] = ph; dwdT[4] = pw; dwdT[5] = 1;
									}
									else if (spatial_transformer_param.type() == SpatialTransformerParameter_TransformType_INVERSE_AFFINE)
									{
										const Dtype *p = params_;
										Dtype det = (p[0] * p[4] - p[1] * p[3]);
										Dtype det2 = det*det;
										// From Matlab sym package: fnc = inv([p0 p1; p3 p4])*[ph - p2; pw - p5];det=(p0*p4 - p1*p3);[simplify(diff(fnc, p0)*det ^ 2); simplify(diff(fnc, p1)*det ^ 2); simplify(diff(fnc, p2)*(p0*p4 - p1*p3) ^ 2); simplify(diff(fnc, p3)*det ^ 2); simplify(diff(fnc, p4)*det^ 2); simplify(diff(fnc, p5)*det ^ 2)]
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
									// This will be similar for other transformations
									for (int param_it = 0; param_it < num_parameters_; param_it++){
										bottom_param_diff[param_it] += tl * -(1 - (pwB - ipwB)) * dhdT[param_it] +
											tl * -(1 - (phB - iphB)) * dwdT[param_it] +
											bl * -(1 - (pwB - ipwB)) * dhdT[param_it] +
											bl *  ((phB - iphB)) * dwdT[param_it] +
											tr *  ((pwB - ipwB)) * dhdT[param_it] +
											tr * -(1 - (phB - iphB)) * dwdT[param_it] +
											br *  ((pwB - ipwB)) * dhdT[param_it] +
											br *  ((phB - iphB)) * dwdT[param_it];
									}
							}
								break;

							default:
								LOG(FATAL) << "Unknown tranform: " << SpatialTransformerParameter_TransformType_INVERSE_AFFINE;
							}
						}
					}
					bottom_data += bottom_->offset(0, 1);
					bottom_diff += bottom_->offset(0, 1);
					top_diff += top_->offset(0, 1);
				}


			}


		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SpatialTransformerLayer);
#endif

	INSTANTIATE_CLASS(SpatialTransformerLayer);
	REGISTER_LAYER_CLASS(SpatialTransformer);

}  // namespace caffe
