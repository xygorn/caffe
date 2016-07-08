#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/spatial_transformer_layer.hpp"





namespace caffe {

	template <typename Dtype> __host__ __device__
		void forwardTransformAffine(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix)
	{
		pdB = pd;
		phB = tfmMatrix[0] * ph + tfmMatrix[1]  * pw + tfmMatrix[2];
		pwB = tfmMatrix[3] * ph + tfmMatrix[4]  * pw + tfmMatrix[5];
	}
	template <typename Dtype>  __host__ __device__
		void dxdTAffine(Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw) 
	{
		dddT[0] = 0;  dddT[1] = 0;  dddT[2] = 0;  
		dddT[3] = 0;  dddT[4] = 0;  dddT[5] = 0;  
		dhdT[0] = ph; dhdT[1] = pw;  dhdT[2] = 1;  
		dhdT[3] = 0;  dhdT[4] = 0;  dhdT[5] = 0;
		dwdT[0] = 0;  dwdT[1] = 0;  dwdT[2] = 0;
		dwdT[3] = ph; dwdT[4] = pw; dwdT[5] = 1;
	}
	template <typename Dtype>  __host__ __device__
		void forwardTransformAffine3D(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix)
	{
		pdB = tfmMatrix[0] * pd + tfmMatrix[1] * ph + tfmMatrix[2]  * pw + tfmMatrix[3];
		phB = tfmMatrix[4] * pd + tfmMatrix[5] * ph + tfmMatrix[6]  * pw + tfmMatrix[7];
		pwB = tfmMatrix[8] * pd + tfmMatrix[9] * ph + tfmMatrix[10] * pw + tfmMatrix[11];
	}
	template <typename Dtype>  __host__ __device__
		void dxdTAffine3D(Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw) 
	{
		dddT[0] = pd; dddT[1] = ph; dddT[2] = pw; dddT[3] = 1;
		dddT[4] = 0;  dddT[5] = 0;  dddT[6] = 0;  dddT[7] = 0;
		dddT[8] = 0;  dddT[9] = 0;  dddT[10]= 0;  dddT[11]= 0;
		dhdT[0] = 0;  dhdT[1] = 0;  dhdT[2] = 0;  dhdT[3] = 0;
		dhdT[4] = pd; dhdT[5] = ph; dhdT[6] = pw; dhdT[7] = 1;
		dhdT[8] = 0;  dhdT[9] = 0;  dhdT[10]= 0;  dhdT[11]= 0;
		dwdT[0] = 0;  dwdT[1] = 0;  dwdT[2] = 0;  dwdT[3] = 0;
		dwdT[4] = 0;  dwdT[5] = 0;  dwdT[6] = 0;  dwdT[7] = 0;
		dwdT[8] = pd; dwdT[9] = ph; dwdT[10]= pw; dwdT[11]= 1;
	}
	template <typename Dtype>  __host__ __device__
		void forwardTransformInverseAffine(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix)
	{
		Dtype det = tfmMatrix[0] * tfmMatrix[4] - tfmMatrix[1] * tfmMatrix[3];
		Dtype inv0 = tfmMatrix[4] / det;
		Dtype inv1 = -tfmMatrix[1]/det;
		Dtype inv3 = -tfmMatrix[3] / det;
		Dtype inv4 = tfmMatrix[0] / det;
		Dtype inv2 = -(inv0 * tfmMatrix[2] + inv1 * tfmMatrix[5]);
		Dtype inv5 = -(inv3 * tfmMatrix[2] + inv4 * tfmMatrix[5]);

		pdB = pd;
		phB = inv0 * ph + inv1 * pw + inv2;
		pwB = inv3 * ph + inv4 * pw + inv5;

	}

	template <typename Dtype>  __host__ __device__
		void dxdTInverseAffine(Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw, const Dtype* tfmMatrix) 
	{
		const Dtype*p=tfmMatrix;
		Dtype det = (p[0] * p[4] - p[1] * p[3]);
		Dtype det2 = det*det;
		dddT[0]=0;dddT[1]=0;dddT[2]=0;dddT[3]=0;dddT[4]=0;dddT[5]=0;
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

	template <typename Dtype> void  precomputeInverseAffine3D(Dtype* dxdT_coeffs_, const Dtype* tfmMatrix) 
	{
	    
		int LUT[9] = {9,2,1, 2,9,0, 1,0,9};
		Dtype det =  tfmMatrix[4*0+0] * tfmMatrix[4*1+1] * tfmMatrix[4*2+2] -tfmMatrix[4*0+0] * tfmMatrix[4*1+2] * tfmMatrix[4*2+1]
					+tfmMatrix[4*0+1] * tfmMatrix[4*1+2] * tfmMatrix[4*2+0] -tfmMatrix[4*0+1] * tfmMatrix[4*1+0] * tfmMatrix[4*2+2]
					+tfmMatrix[4*0+2] * tfmMatrix[4*1+0] * tfmMatrix[4*2+1] -tfmMatrix[4*0+2] * tfmMatrix[4*1+1] * tfmMatrix[4*2+0];
		Dtype detUV[9];
		detUV[3*0+0] = tfmMatrix[4*1+1]*tfmMatrix[4*2+2]-tfmMatrix[4*1+2]*tfmMatrix[4*2+1];
		detUV[3*0+1] = tfmMatrix[4*1+0]*tfmMatrix[4*2+2]-tfmMatrix[4*1+2]*tfmMatrix[4*2+0];
		detUV[3*0+2] = tfmMatrix[4*1+0]*tfmMatrix[4*2+1]-tfmMatrix[4*1+1]*tfmMatrix[4*2+0];
		detUV[3*1+0] = tfmMatrix[4*0+1]*tfmMatrix[4*2+2]-tfmMatrix[4*0+2]*tfmMatrix[4*2+1];
		detUV[3*1+1] = tfmMatrix[4*0+0]*tfmMatrix[4*2+2]-tfmMatrix[4*0+2]*tfmMatrix[4*2+0];
		detUV[3*1+2] = tfmMatrix[4*0+0]*tfmMatrix[4*2+1]-tfmMatrix[4*0+1]*tfmMatrix[4*2+0];
		detUV[3*2+0] = tfmMatrix[4*0+1]*tfmMatrix[4*1+2]-tfmMatrix[4*0+2]*tfmMatrix[4*1+1];
		detUV[3*2+1] = tfmMatrix[4*0+0]*tfmMatrix[4*1+2]-tfmMatrix[4*0+2]*tfmMatrix[4*1+0];
		detUV[3*2+2] = tfmMatrix[4*0+0]*tfmMatrix[4*1+1]-tfmMatrix[4*0+1]*tfmMatrix[4*1+0];

		for (int pRIt = 0;pRIt<3;++pRIt) {
			for (int pUIt = 0;pUIt<3;++pUIt) {// X = d h w
				for (int pCIt = 0;pCIt<3;++pCIt) {
					for (int pVIt = 0;pVIt<3;++pVIt) { // pd ph and pw coeffs
						// dFnc(U)dT() = [ a/det - (c*detVU*(-1)^(V+U))/det^2 ]  * [pd;ph;pw;1]
						int W = LUT[3*pRIt+pVIt];
						int Q = LUT[3*pCIt+pUIt];
						Dtype a = ((pUIt==pCIt||pVIt==pRIt)?0: tfmMatrix[4*W+Q]*((((pUIt-pCIt+3)%3)+((pVIt-pRIt+3)%3))%2?-1:1)   );
						Dtype c = detUV[3*pRIt+pCIt]*((pRIt+pCIt)%2?-1:1);
						dxdT_coeffs_[12*(4*pRIt+pCIt)+4*pUIt+pVIt]=   a/det - (c*detUV[3*pVIt+pUIt]*((pVIt+pUIt)%2?-1:1))/det/det;// coord coeffs for dSkewRotation
						dxdT_coeffs_[12*(4*pRIt+3)+4*pUIt+pVIt]=0; // coord coeffs for dTranslation
					}
					dxdT_coeffs_[12*(4*pRIt+pCIt)+4*pUIt+3]=-tfmMatrix[4*0+3]*dxdT_coeffs_[12*(4*pRIt+pCIt)+4*pUIt+0]
					                                        -tfmMatrix[4*1+3]*dxdT_coeffs_[12*(4*pRIt+pCIt)+4*pUIt+1]
															-tfmMatrix[4*2+3]*dxdT_coeffs_[12*(4*pRIt+pCIt)+4*pUIt+2]; // scalar coeffs for dSkewRotation
				}
				dxdT_coeffs_[12*(4*pRIt+3)+4*pUIt+3]=-detUV[3*pRIt+pUIt]/det * ((pRIt+pUIt)%2?-1:1); // scalar coeffs for dTranslation
			}
		}

	}

		template <typename Dtype>  __host__ __device__
		void forwardTransformInverseAffine3D(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix)
		{
			Dtype det =  tfmMatrix[4*0+0] * tfmMatrix[4*1+1] * tfmMatrix[4*2+2] -tfmMatrix[4*0+0] * tfmMatrix[4*1+2] * tfmMatrix[4*2+1]
						+tfmMatrix[4*0+1] * tfmMatrix[4*1+2] * tfmMatrix[4*2+0] -tfmMatrix[4*0+1] * tfmMatrix[4*1+0] * tfmMatrix[4*2+2]
						+tfmMatrix[4*0+2] * tfmMatrix[4*1+0] * tfmMatrix[4*2+1] -tfmMatrix[4*0+2] * tfmMatrix[4*1+1] * tfmMatrix[4*2+0];  

			Dtype detUV[9];
			detUV[3*0+0] = tfmMatrix[4*1+1]*tfmMatrix[4*2+2]-tfmMatrix[4*1+2]*tfmMatrix[4*2+1];
			detUV[3*0+1] = tfmMatrix[4*1+0]*tfmMatrix[4*2+2]-tfmMatrix[4*1+2]*tfmMatrix[4*2+0];
			detUV[3*0+2] = tfmMatrix[4*1+0]*tfmMatrix[4*2+1]-tfmMatrix[4*1+1]*tfmMatrix[4*2+0];
			detUV[3*1+0] = tfmMatrix[4*0+1]*tfmMatrix[4*2+2]-tfmMatrix[4*0+2]*tfmMatrix[4*2+1];
			detUV[3*1+1] = tfmMatrix[4*0+0]*tfmMatrix[4*2+2]-tfmMatrix[4*0+2]*tfmMatrix[4*2+0];
			detUV[3*1+2] = tfmMatrix[4*0+0]*tfmMatrix[4*2+1]-tfmMatrix[4*0+1]*tfmMatrix[4*2+0];
			detUV[3*2+0] = tfmMatrix[4*0+1]*tfmMatrix[4*1+2]-tfmMatrix[4*0+2]*tfmMatrix[4*1+1];
			detUV[3*2+1] = tfmMatrix[4*0+0]*tfmMatrix[4*1+2]-tfmMatrix[4*0+2]*tfmMatrix[4*1+0];
			detUV[3*2+2] = tfmMatrix[4*0+0]*tfmMatrix[4*1+1]-tfmMatrix[4*0+1]*tfmMatrix[4*1+0];

			pdB =( detUV[3*0+0] * (pd-tfmMatrix[4*0+3]) + detUV[3*1+0]*(ph-tfmMatrix[4*1+3]) + detUV[3*2+0]*(pw-tfmMatrix[4*2+3]) )/det;
			phB =( detUV[3*0+1] * (pd-tfmMatrix[4*0+3]) + detUV[3*1+1]*(ph-tfmMatrix[4*1+3]) + detUV[3*2+1]*(pw-tfmMatrix[4*2+3]) )/det;
			pwB =( detUV[3*0+2] * (pd-tfmMatrix[4*0+3]) + detUV[3*1+2]*(ph-tfmMatrix[4*1+3]) + detUV[3*2+2]*(pw-tfmMatrix[4*2+3]) )/det;

		}

	template <typename Dtype>  __host__ __device__
		void dxdTInverseAffine3D(const Dtype* dxdT_coeffs,Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw) 
	{
		for (int paramIt=0;paramIt<1;++paramIt) {
			dddT[paramIt]=dxdT_coeffs[12*paramIt+4*0+0]*pd + dxdT_coeffs[12*paramIt+4*0+1]*ph + dxdT_coeffs[12*paramIt+4*0+2]*pw + dxdT_coeffs[12*paramIt+4*0+3];
			dhdT[paramIt]=dxdT_coeffs[12*paramIt+4*1+0]*pd + dxdT_coeffs[12*paramIt+4*1+1]*ph + dxdT_coeffs[12*paramIt+4*1+2]*pw + dxdT_coeffs[12*paramIt+4*1+3];
			dwdT[paramIt]=dxdT_coeffs[12*paramIt+4*2+0]*pd + dxdT_coeffs[12*paramIt+4*2+1]*ph + dxdT_coeffs[12*paramIt+4*2+2]*pw + dxdT_coeffs[12*paramIt+4*2+3];
		}
	}

	template <typename Dtype>  __host__ __device__
		void forwardInterpolate(Dtype* top_data, const Dtype* bottom_data, int index, int depth, int height, int width, Dtype pdB, Dtype phB, Dtype pwB)
	{
		int ipdB, ipwB, iphB;
		Dtype tln, bln, trn, brn, tlf, blf, trf, brf;
			
			pdB = max(0., min(static_cast<Dtype>(depth - 1), pdB));
			phB = max(0., min(static_cast<Dtype>(height - 1), phB));
			pwB = max(0., min(static_cast<Dtype>(width - 1), pwB));

			ipwB = floor(pwB);
			iphB = floor(phB);
			ipdB = floor(pdB);

			tln = bottom_data[ipdB*width*height + iphB*width + ipwB]; 
			bln = bottom_data[ipdB*width*height + (iphB + 1)*width + ipwB];
			trn = bottom_data[ipdB*width*height + iphB*width + (ipwB + 1)];
			brn = bottom_data[ipdB*width*height + (iphB + 1)*width + (ipwB + 1)];
			tlf = bottom_data[(ipdB+1)*width*height + iphB*width + ipwB];
			blf = bottom_data[(ipdB+1)*width*height + (iphB + 1)*width + ipwB];
			trf = bottom_data[(ipdB+1)*width*height + iphB*width + (ipwB + 1)];
			brf = bottom_data[(ipdB+1)*width*height + (iphB + 1)*width + (ipwB + 1)];
			top_data[index] = tln*(1 - (pdB - ipdB)) * (1 - (phB - iphB)) * (1 - (pwB - ipwB)) +
			    bln*(1 - (pdB - ipdB)) *(    (phB - iphB)) * (1 - (pwB - ipwB)) +
			    trn*(1 - (pdB - ipdB)) *(1 - (phB - iphB)) * ((pwB - ipwB)) +
			    brn*(1 - (pdB - ipdB)) *((phB - iphB)) * ((pwB - ipwB)) +
			    tlf*((pdB - ipdB)) *(1 - (phB - iphB)) * (1 - (pwB - ipwB)) +
			    blf*((pdB - ipdB)) *(    (phB - iphB)) * (1 - (pwB - ipwB)) +
			    trf*((pdB - ipdB)) *(1 - (phB - iphB)) * ((pwB - ipwB)) +
			    brf*((pdB - ipdB)) *((phB - iphB)) * ((pwB - ipwB));
	}


	template <typename Dtype>
	__global__ void AffineForward(const int nthreads, const Dtype* bottom_data,
		const int num, const int channels, const int depth, const int height,
		const int width, const Dtype* tfmMatrix, const int num_parameters,
		Dtype* top_data, const int grid_d, const int grid_h, const int grid_w, SpatialTransformerParameter_TransformType transform_type) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Dtype pdB, pwB, phB;
			int pw = index % grid_w;
			int ph = (index / grid_w) % grid_h;
			int pd = (index / (grid_w * grid_h)) % grid_d;
			int c = (index / (grid_w * grid_h * grid_d)) % channels;
			int n = index / (grid_w * grid_h * grid_d * channels);

			bottom_data += (n * channels + c) * height * width * depth;
			tfmMatrix += n*num_parameters;
			switch (transform_type) {
			case SpatialTransformerParameter_TransformType_AFFINE:
				forwardTransformAffine(pdB, phB, pwB, pd, ph, pw, tfmMatrix);
				break;
			case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
				forwardTransformInverseAffine(pdB, phB, pwB, pd, ph, pw, tfmMatrix);
				break;
			case SpatialTransformerParameter_TransformType_AFFINE3D:
				forwardTransformAffine3D(pdB, phB, pwB, pd, ph, pw, tfmMatrix);
				break;
			case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
				forwardTransformInverseAffine3D(pdB, phB, pwB, pd, ph, pw, tfmMatrix);
				break;
			}

			// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
			forwardInterpolate(top_data, bottom_data, index, depth, height, width, pdB, phB, pwB);

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

			int tDepth;
			if (bottom_->num_axes()==4)
			{ tDepth=1; }
			else
			{ tDepth=bottom_->shape(-3); }

			const int depth = tDepth;
			const int width = bottom_->shape(-1);
			const int height = bottom_->shape(-2);


			int count = top_->count();


			// NOLINT_NEXT_LINE(whitespace/operators)
			AffineForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data, bottom_->shape(0), bottom_->shape(1),
				depth, height, width, params, num_parameters_, top_data, grid_d_, grid_h_, grid_w_, spatial_transformer_param.type());
			CUDA_POST_KERNEL_CHECK;

		}
	}


	template <typename Dtype>
	__global__ void AffineBackward(const int nthreads, const Dtype* top_diff,
		const int num, const int channels, const int grid_d, const int grid_h, const int grid_w,
		const Dtype * tfmMatrix, Dtype* param_diff, const Dtype* buffer, const int num_parameters,
		const Dtype* bottom_data, Dtype* bottom_diff, const int depth, const int height, const int width, SpatialTransformerParameter_TransformType transform_type, bool propagate_down_data, bool propagate_down_param) {

		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the grid index
			Dtype dddT[12]; // temporary storage for partial derivatives with respect to transform parameters
			Dtype dhdT[12]; // Hack currently set for largest possible number of parameters
			Dtype dwdT[12];
			int pw = index % grid_w;
			int ph = (index / grid_w) % grid_h;
			int pd = (index / (grid_w * grid_h)) % grid_d;
			int c = (index / (grid_w * grid_h * grid_d)) % channels;
			int n = index / (grid_w * grid_h * grid_d * channels);
			Dtype pwB, phB, pdB, tln, bln, trn, brn, tlf, blf, trf, brf;
			int ipwB, iphB, ipdB;

			top_diff += (n * channels + c) * grid_h * grid_w * grid_d;
			bottom_diff += (n * channels + c) * height * width * depth;
			bottom_data += (n * channels + c) * height * width * depth;
			tfmMatrix += num_parameters*n;
			param_diff += num_parameters*n;
			switch (transform_type) {
			case SpatialTransformerParameter_TransformType_AFFINE3D:
				forwardTransformAffine3D(pdB,phB,pwB, pd, ph, pw, tfmMatrix);
			break;
			case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
				forwardTransformInverseAffine3D(pdB,phB,pwB, pd, ph, pw, tfmMatrix);
			break;
			case SpatialTransformerParameter_TransformType_AFFINE:
				forwardTransformAffine(pdB,phB,pwB, pd, ph, pw, tfmMatrix);
			break;
			case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
				forwardTransformInverseAffine(pdB,phB,pwB, pd, ph, pw, tfmMatrix);
			break;
			}
			
			// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
			pdB = max(0., min(static_cast<Dtype>(depth - 1), pdB));
			phB = max(0., min(static_cast<Dtype>(height - 1), phB));
			pwB = max(0., min(static_cast<Dtype>(width - 1), pwB));

			// This will be similar for other transformation (with same sampling kernel)
			ipwB = floor(pwB);
			iphB = floor(phB);
			ipdB = floor(pdB);
			if (propagate_down_data)
			{
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (1-(pdB-ipdB)) * (1-(phB-iphB)) * (1 - (pwB-ipwB)), bottom_diff +  ipdB     *width*height +  iphB     *width +  ipwB);
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (1-(pdB-ipdB)) * (1-(phB-iphB)) * (    (pwB-ipwB)), bottom_diff +  ipdB     *width*height +  iphB     *width + (ipwB + 1));
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (1-(pdB-ipdB)) * (  (phB-iphB)) * (1 - (pwB-ipwB)), bottom_diff +  ipdB     *width*height + (iphB + 1)*width +  ipwB);
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (1-(pdB-ipdB)) * (  (phB-iphB)) * (    (pwB-ipwB)), bottom_diff +  ipdB     *width*height + (iphB + 1)*width + (ipwB + 1));
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (  (pdB-ipdB)) * (1-(phB-iphB)) * (1 - (pwB-ipwB)), bottom_diff + (ipdB + 1)*width*height +  iphB     *width +  ipwB);
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (  (pdB-ipdB)) * (1-(phB-iphB)) * (    (pwB-ipwB)), bottom_diff + (ipdB + 1)*width*height +  iphB     *width + (ipwB + 1));
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (  (pdB-ipdB)) * (  (phB-iphB)) * (1 - (pwB-ipwB)), bottom_diff + (ipdB + 1)*width*height + (iphB + 1)*width +  ipwB);
				caffe_gpu_atomic_add(top_diff[ph*grid_w + pw] * (  (pdB-ipdB)) * (  (phB-iphB)) * (    (pwB-ipwB)), bottom_diff + (ipdB + 1)*width*height + (iphB + 1)*width + (ipwB + 1));
			}
			if (propagate_down_param)
			{
                  tln = bottom_data[ipdB*height*width + iphB*width + ipwB];
                  bln = bottom_data[ipdB*height*width + (iphB + 1)*width + ipwB];
                  trn = bottom_data[ipdB*height*width + iphB*width + (ipwB + 1)];
                  brn = bottom_data[ipdB*height*width + (iphB + 1)*width + (ipwB + 1)];
                  tlf = bottom_data[(ipdB+1)*height*width + iphB*width + ipwB];
                  blf = bottom_data[(ipdB+1)*height*width + (iphB + 1)*width + ipwB];
                  trf = bottom_data[(ipdB+1)*height*width + iphB*width + (ipwB + 1)];
                  brf = bottom_data[(ipdB+1)*height*width + (iphB + 1)*width + (ipwB + 1)];

				// This depends on the transformation function:
				switch (transform_type) {
				case SpatialTransformerParameter_TransformType_AFFINE3D:
					dxdTAffine3D(dddT, dhdT, dwdT, pd, ph, pw);
				break;
				case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
					dxdTInverseAffine3D(buffer+n*144, dddT, dhdT, dwdT, pd, ph, pw);
				break;
				case SpatialTransformerParameter_TransformType_AFFINE:
					dxdTAffine(dddT, dhdT, dwdT, pd, ph, pw);
				break;
				case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
					dxdTInverseAffine3D(buffer, dddT, dhdT, dwdT, pd, ph, pw);
				break;
				}

				// This will be similar for other transformations (except with all partial derivatives)
				for (int param_it = 0; param_it < num_parameters; param_it++){
					caffe_gpu_atomic_add(  dddT[param_it] * (tlf*(iphB - phB + 1)*(ipwB - pwB + 1) - tln*(iphB - phB + 1)*(ipwB - pwB + 1) - trf*(ipwB - pwB)*(iphB - phB + 1) + trn*(ipwB - pwB)*(iphB - phB + 1) - blf*(iphB - phB)*(ipwB - pwB + 1) + bln*(iphB - phB)*(ipwB - pwB + 1) + brf*(iphB - phB)*(ipwB - pwB) - brn*(iphB - phB)*(ipwB - pwB))
					                     + dhdT[param_it] * (tlf*(ipdB - pdB)*(ipwB - pwB + 1) - tln*(ipdB - pdB + 1)*(ipwB - pwB + 1) - trf*(ipdB - pdB)*(ipwB - pwB) + trn*(ipwB - pwB)*(ipdB - pdB + 1) - blf*(ipdB - pdB)*(ipwB - pwB + 1) + bln*(ipdB - pdB + 1)*(ipwB - pwB + 1) + brf*(ipdB - pdB)*(ipwB - pwB) - brn*(ipwB - pwB)*(ipdB - pdB + 1))
					                     + dwdT[param_it] * (tlf*(ipdB - pdB)*(iphB - phB + 1) - tln*(ipdB - pdB + 1)*(iphB - phB + 1) - trf*(ipdB - pdB)*(iphB - phB + 1) + trn*(ipdB - pdB + 1)*(iphB - phB + 1) - blf*(ipdB - pdB)*(iphB - phB) + bln*(iphB - phB)*(ipdB - pdB + 1) + brf*(ipdB - pdB)*(iphB - phB) - brn*(iphB - phB)*(ipdB - pdB + 1))
					                    , param_diff+param_it);
				}
			}
		}
	}



	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();

		const Dtype* params_cpu;
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

			//Precomputation
			const Dtype* buffer = 0;
			switch (spatial_transformer_param.type()) {
			case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
				if (hasConstParams)
				{
					params_cpu= constParamsBlob_.cpu_data();
				}
				else
				{
					params_cpu= bottom[0]->cpu_data();
				}
				for (int n = 0; n < bottom_->num(); ++n) {
					precomputeInverseAffine3D((buffer_[0]->mutable_cpu_diff())+144*n, params_cpu + n*num_parameters_) ;
				}
				buffer = buffer_[0]->gpu_data(); 
			}


			
			int tDepth;
			if (bottom_->num_axes()==4)
			{ tDepth=1; }
			else
			{ tDepth=bottom_->shape(-3); }

			const int depth = tDepth;
			const int width = bottom_->shape(-1);
			const int height = bottom_->shape(-2);


		// NOLINT_NEXT_LINE(whitespace/operators)
		AffineBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, top_diff, top_->num(), top_->channels(),
			grid_d_, grid_h_, grid_w_, params, param_diff, buffer, num_parameters_, bottom_data, bottom_diff,
			depth, height, width, spatial_transformer_param.type(), propagate_down[i], propagate_down[0] && !hasConstParams);
		CUDA_POST_KERNEL_CHECK;
		}
	}


		INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);


}  // namespace caffe
