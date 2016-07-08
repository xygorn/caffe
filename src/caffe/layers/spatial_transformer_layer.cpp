#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {
	template <typename Dtype> __host__ __device__
		void forwardTransformAffine(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix);
	template <typename Dtype>  __host__ __device__
		void dxdTAffine(Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw) ;
	template <typename Dtype>  __host__ __device__
		void forwardTransformAffine3D(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix);
	template <typename Dtype>  __host__ __device__
		void dxdTAffine3D(Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw) ;
	template <typename Dtype>  __host__ __device__
		void forwardTransformInverseAffine(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix);
	template <typename Dtype>  __host__ __device__
		void dxdTInverseAffine(Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw) ;
	template <typename Dtype> void  precomputeInverseAffine3D(Dtype* dxdT_coeffs_, Dtype* tfmMatrix_) ;
			template <typename Dtype>  __host__ __device__
		void forwardTransformInverseAffine3D(Dtype& pdB,Dtype& phB,Dtype& pwB,int  pd,int  ph,int pw, const Dtype* tfmMatrix);
	template <typename Dtype>  __host__ __device__
		void dxdTInverseAffine3D(const Dtype* dxdT_coeffs,Dtype* dddT,Dtype* dhdT,Dtype* dwdT,int pd, int ph, int pw) ;
	template <typename Dtype>  __host__ __device__
		void forwardInterpolate(Dtype* top_data, const Dtype* bottom_data, int index, int depth, int height, int width, Dtype pdB, Dtype phB, Dtype pwB);
	template <typename Dtype> void  precomputeInverseAffine3D(Dtype* dxdT_coeffs_, const Dtype* tfmMatrix_) ;

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
			
			pdB = std::max(static_cast<Dtype>(0.), std::min(static_cast<Dtype>(depth - 1), pdB));
			phB = std::max(static_cast<Dtype>(0.), std::min(static_cast<Dtype>(height - 1), phB));
			pwB = std::max(static_cast<Dtype>(0.), std::min(static_cast<Dtype>(width - 1), pwB));

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
	void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Configure the kernel size, padding, stride, and inputs.
		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();

		CHECK(! ((spatial_transformer_param.has_grid_h() || spatial_transformer_param.has_grid_w()) && (spatial_transformer_param.grid_size().size()>0))) << "Specify either grid_w and grid_h or grid_size.";
		CHECK( spatial_transformer_param.grid_size().size() <= 3) << "More than 3 dimensions not supported; try reshaping the layer to treat higher dimensions as channels";
		switch ( spatial_transformer_param.grid_size().size()) {
		case 0:
			CHECK( (spatial_transformer_param.has_grid_h() && spatial_transformer_param.has_grid_w()) ) << "Specify either grid_w and grid_h or grid_size.";
			grid_d_=1;
			grid_h_=spatial_transformer_param.grid_h();
			grid_w_=spatial_transformer_param.grid_w();
			break;
		case 1:
			grid_d_=spatial_transformer_param.grid_size(0);
			grid_h_=spatial_transformer_param.grid_size(0);
			grid_w_=spatial_transformer_param.grid_size(0);
			break;
		case 2:
			grid_d_=1;
			grid_h_=spatial_transformer_param.grid_size(0);
			grid_w_=spatial_transformer_param.grid_size(0);
		case 3:
			grid_d_=spatial_transformer_param.grid_size(0);
			grid_h_=spatial_transformer_param.grid_size(1);
			grid_w_=spatial_transformer_param.grid_size(2);
		}
        CHECK_GT(grid_d_, 0) << "Resampling grid dimensions must be positive integers."; 
        CHECK_GT(grid_h_, 0) << "Resampling grid dimensions must be positive integers.";
		CHECK_GT(grid_w_, 0) << "Resampling grid dimensions must be positive integers.";

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
		
		vector<int> bufferShape(4);// only used in inverseaffine3d, but moved here to avoid compilation errors
		switch (spatial_transformer_param.type())
		{
		case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
			if (this->buffer_.size() > 0) {
				LOG(INFO) << "Skipping parameter initialization";
			} else {
				bufferShape[3]=1;
				bufferShape[2]=1;
				bufferShape[1]=144;
				bufferShape[0]=bottom[0]->shape(0);
				buffer_.push_back(new Blob<Dtype>(bufferShape));
			}

			num_parameters_ = 12;
			break;
		case SpatialTransformerParameter_TransformType_AFFINE3D:
			num_parameters_ = 12;
			break;
		case SpatialTransformerParameter_TransformType_AFFINE:
		case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
			num_parameters_ = 6;
			break;
		}
		if (spatial_transformer_param.const_params().size())
		{
			CHECK_EQ(spatial_transformer_param.const_params().size(), num_parameters_) << "Number of constant parameters must match number of parameters in transform type.";
		}
		else
		{
			CHECK_EQ(bottom[0]->channels(), num_parameters_) << "Number of channels in first input must match number of parameters in transform type.";
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

		Dtype pwB, phB,pdB;
		for (int i = firstDataBlob; i < bottom.size(); ++i) {
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - firstDataBlob];
			const Dtype* bottom_data = bottom_->cpu_data();
			Dtype* top_data = top_->mutable_cpu_data();
			int tDepth;
			if (bottom_->num_axes()==4)
			{ tDepth=1; }
			else
			{ tDepth=bottom_->shape(-3); }

			const int depth = tDepth;
			const int width = bottom_->shape(-1);
			const int height = bottom_->shape(-2);
			caffe_set(top_->count(), Dtype(0), top_data);
			// The main loop
			for (int n = 0; n < bottom_->num(); ++n) {
				const Dtype* params_ = params + n*num_parameters_;

				for(int c = 0; c < bottom_->channels(); ++c) {
					for (int pd = 0; pd < grid_d_; ++pd) {
						for (int ph = 0; ph < grid_h_; ++ph) {
							for (int pw = 0; pw < grid_w_; ++pw) {
								const int index = pd * grid_w_ * grid_h_ + ph * grid_w_ + pw;
								// transform point will be [pdB,phB,pwB]=transformPoint(pd,ph,pw)
								// TODO move the pixel loop outside of the rest to avoid repeatedly calculating the weights
								switch (spatial_transformer_param.type()) { 
								case SpatialTransformerParameter_TransformType_AFFINE3D:
									forwardTransformAffine3D(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
									forwardTransformInverseAffine3D(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								case SpatialTransformerParameter_TransformType_AFFINE:
									forwardTransformAffine(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
									forwardTransformInverseAffine(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								}
								// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
								forwardInterpolate(top_data, bottom_data, index, depth, height, width, pdB, phB, pwB);
							}
						}
						bottom_data += bottom_->offset(0, 1);

						top_data += top_->offset(0, 1);
					}
				}
							

						
			}
					
		}


	}

	template <typename Dtype>
	void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		SpatialTransformerParameter spatial_transformer_param = this->layer_param_.spatial_transformer_param();
		Dtype dddT[12]; // temporary storage for partial derivatives with respect to transform parameters
		Dtype dhdT[12]; // should be fixed to be of size num_parameters_
		Dtype dwdT[12];
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

		

		Dtype pdB, pwB, phB, tln, bln, trn, brn, tlf, blf, trf, brf;
		int ipdB, ipwB, iphB;
		for (int i = firstDataBlob; i < bottom.size(); ++i) {
			Blob<Dtype>* bottom_ = bottom[i];
			Blob<Dtype>* top_ = top[i - firstDataBlob];

			const Dtype* top_diff = top_->cpu_diff();
			Dtype* bottom_diff = bottom_->mutable_cpu_diff();
			const Dtype* bottom_data = bottom_->mutable_cpu_data();

			int tDepth;
			if (bottom_->num_axes()==4)
			{ tDepth=1; }
			else
			{ tDepth=bottom_->shape(-3); }

			const int depth = tDepth;
			const int width = bottom_->shape(-1);
			const int height = bottom_->shape(-2);

			caffe_set(bottom_->count(), Dtype(0), bottom_diff);
			for (int n = 0; n < bottom_->num(); ++n) {
				const Dtype* params_ = params + n*num_parameters_;

				switch (spatial_transformer_param.type()) {
				case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
					precomputeInverseAffine3D((buffer_[0]->mutable_cpu_diff())+144*n, params_); 
				break;
				}

				for (int c = 0; c < bottom_->channels(); ++c) {
					for (int pd = 0; pd < grid_d_; ++pd) {
						for (int ph = 0; ph < grid_h_; ++ph) {
							for (int pw = 0; pw < grid_w_; ++pw) {
								const int index = ph * grid_w_ + pw;
								// transform point will be [phB,pwB]=transformPoint(ph,pw)
								// TODO move the pixel loop outside of the rest to avoid repeatedly calculating the weights
								switch (spatial_transformer_param.type()) {
								case SpatialTransformerParameter_TransformType_AFFINE3D:
									forwardTransformAffine(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								case SpatialTransformerParameter_TransformType_INVERSE_AFFINE3D:
									forwardTransformInverseAffine(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								case SpatialTransformerParameter_TransformType_AFFINE:
									forwardTransformAffine(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
									forwardTransformInverseAffine(pdB,phB,pwB, pd, ph, pw, params_);
								break;
								}

									// TODO genericize boundary handling - for now assume smooth boundary extension (extend boundary value outside the image)
								pdB = std::max((Dtype)0., std::min(static_cast<Dtype>(depth - 1), pdB));
								phB = std::max((Dtype)0., std::min(static_cast<Dtype>(height - 1), phB));
								pwB = std::max((Dtype)0., std::min(static_cast<Dtype>(width - 1), pwB));

								ipdB = floor(pdB);
								ipwB = floor(pwB);
								iphB = floor(phB);
								bottom_diff[ ipdB   *height*width + iphB   *width +  ipwB]   += 1+top_diff[index] * (1-(pdB - ipdB)) * (1-(phB - iphB)) * (1-(pwB - ipwB));
								bottom_diff[ ipdB   *height*width + iphB   *width + (ipwB+1)]+= top_diff[index] * (1-(pdB - ipdB)) * (1-(phB - iphB)) * (  (pwB - ipwB));
								bottom_diff[ ipdB   *height*width +(iphB+1)*width +  ipwB]   += top_diff[index] * (1-(pdB - ipdB)) * (  (phB - iphB)) * (1-(pwB - ipwB));
								bottom_diff[ ipdB   *height*width +(iphB+1)*width + (ipwB+1)]+=	top_diff[index] * (1-(pdB - ipdB)) * (  (phB - iphB)) * (  (pwB - ipwB));
								bottom_diff[(ipdB+1)*height*width + iphB   *width + ipwB]    += top_diff[index] * (  (pdB - ipdB)) * (1-(phB - iphB)) * (1-(pwB - ipwB));
								bottom_diff[(ipdB+1)*height*width + iphB   *width + (ipwB+1)]+= top_diff[index] * (  (pdB - ipdB)) * (1-(phB - iphB)) * (  (pwB - ipwB));
								bottom_diff[(ipdB+1)*height*width +(iphB+1)*width + ipwB]    += top_diff[index] * (  (pdB - ipdB)) * (  (phB - iphB)) * (1-(pwB - ipwB));
								bottom_diff[(ipdB+1)*height*width +(iphB+1)*width + (ipwB+1)]+= top_diff[index] * (  (pdB - ipdB)) * (  (phB - iphB)) * (  (pwB - ipwB));
								if (!spatial_transformer_param.const_params().size())
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
									switch(spatial_transformer_param.type()) {
									case  SpatialTransformerParameter_TransformType_AFFINE:
										dxdTAffine(dddT,dhdT,dwdT,pd, ph, pw);
									break;
									case SpatialTransformerParameter_TransformType_INVERSE_AFFINE:
										dxdTInverseAffine(dddT,dhdT,dwdT,pd, ph, pw,params_);
									break;
									}
									// This will be similar for other transformations
									for (int param_it = 0; param_it < num_parameters_; param_it++){
										bottom_param_diff[param_it] +=  dddT[param_it] * (tlf*(iphB - phB + 1)*(ipwB - pwB + 1) - tln*(iphB - phB + 1)*(ipwB - pwB + 1) - trf*(ipwB - pwB)*(iphB - phB + 1) + trn*(ipwB - pwB)*(iphB - phB + 1) - blf*(iphB - phB)*(ipwB - pwB + 1) + bln*(iphB - phB)*(ipwB - pwB + 1) + brf*(iphB - phB)*(ipwB - pwB) - brn*(iphB - phB)*(ipwB - pwB));
										bottom_param_diff[param_it] +=  dhdT[param_it] * (tlf*(ipdB - pdB)*(ipwB - pwB + 1) - tln*(ipdB - pdB + 1)*(ipwB - pwB + 1) - trf*(ipdB - pdB)*(ipwB - pwB) + trn*(ipwB - pwB)*(ipdB - pdB + 1) - blf*(ipdB - pdB)*(ipwB - pwB + 1) + bln*(ipdB - pdB + 1)*(ipwB - pwB + 1) + brf*(ipdB - pdB)*(ipwB - pwB) - brn*(ipwB - pwB)*(ipdB - pdB + 1));
										bottom_param_diff[param_it] +=  dwdT[param_it] * (tlf*(ipdB - pdB)*(iphB - phB + 1) - tln*(ipdB - pdB + 1)*(iphB - phB + 1) - trf*(ipdB - pdB)*(iphB - phB + 1) + trn*(ipdB - pdB + 1)*(iphB - phB + 1) - blf*(ipdB - pdB)*(iphB - phB) + bln*(iphB - phB)*(ipdB - pdB + 1) + brf*(ipdB - pdB)*(iphB - phB) - brn*(iphB - phB)*(ipdB - pdB + 1));

									}
								}
								
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
