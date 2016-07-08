#ifndef CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_
#define CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include <boost/shared_array.hpp>

namespace caffe {

/**
* @brief Applies parameterized transforms (i.e. the grid generator and resampler
*        in Spatial Transformer Networks). 
*/

template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {
public:
	/**
	* @param param provides SpatialTransformerParameter spatial_transformer_param,
	*    with SpatialTransformerLayer options:
	*  - grid_size / grid_h / grid_w. The resampling grid dimensions, given by
	*  grid_size for square grids or grid_h and grid_w for rectangular
	*  grids.
	*  - transform_type. The type of transform to use. Only Affine is  currently supported.
	*/
	explicit SpatialTransformerLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SpatialTransformer"; }
	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 0; }
	virtual inline bool EqualNumBottomTopBlobs() const { return false; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	//bool AllowForceBackward(const int bottom_index);
	Blob<Dtype> constParamsBlob_;
	vector<Blob<Dtype> > buffer_;
	int grid_h_, grid_w_, grid_d_;
	int num_parameters_;

};

}  // namespace caffe

#endif  // CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_
