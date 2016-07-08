#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace caffe {

template <typename Dtype>
void HDF5OutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  file_name_ = this->layer_param_.hdf5_output_param().file_name();
  file_id_ = H5Fcreate(file_name_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  CHECK_GE(file_id_, 0) << "Failed to open HDF5 file" << file_name_;
  iter_ = 0;
  file_opened_ = true;
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
  if (file_opened_) {
    herr_t status = H5Fclose(file_id_);
    CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
  }
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SaveBlobs() {
  // TODO: no limit on the number of blobs
  LOG(INFO) << "Saving HDF5 file " << file_name_ << " iteration " << iter_;
  CHECK_EQ(data_blob_.shape(0), label_blob_.shape(0)) <<
      "data blob and label blob must have the same batch size";
  ostringstream dataset_name;
  dataset_name << HDF5_DATA_DATASET_NAME << iter_;
  hdf5_save_nd_dataset(file_id_, dataset_name.str(), data_blob_);
  dataset_name.str("");
  dataset_name << HDF5_DATA_LABEL_NAME << iter_;
  hdf5_save_nd_dataset(file_id_, dataset_name.str(), label_blob_);
  LOG(INFO) << "Successfully saved " << data_blob_.shape(0) << " rows";
  iter_++;
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom.size(), 2);
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  data_blob_.Reshape(bottom[0]->shape());
  label_blob_.Reshape(bottom[1]->shape());
  const int data_datum_dim = bottom[0]->count() / bottom[0]->shape(0);
  const int label_datum_dim = bottom[1]->count() / bottom[1]->shape(0);

  for (int i = 0; i < bottom[0]->shape(0); ++i) {
    caffe_copy(data_datum_dim, &bottom[0]->cpu_data()[i * data_datum_dim],
        &data_blob_.mutable_cpu_data()[i * data_datum_dim]);
    caffe_copy(label_datum_dim, &bottom[1]->cpu_data()[i * label_datum_dim],
        &label_blob_.mutable_cpu_data()[i * label_datum_dim]);
  }
  SaveBlobs();
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(HDF5OutputLayer);
#endif

INSTANTIATE_CLASS(HDF5OutputLayer);
REGISTER_LAYER_CLASS(HDF5Output);

}  // namespace caffe
