// Copyright (c) OpenMMLab. All rights reserved.
#include "trt_bev_pool.hpp"

#include <assert.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdint>

#include "trt_bev_pool_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"
#include <assert.h>

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"bev_pool_v2"};
}  // namespace

TRTBEVPoolV2::TRTBEVPoolV2(const std::string &name, int32_t outWidth, int32_t outHeight, int32_t outZ) :
  TRTPluginBase(name),
  mOutWidth(outWidth),
  mOutHeight(outHeight),
  mOutZ(outZ){}

TRTBEVPoolV2::TRTBEVPoolV2(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mOutWidth);
  deserialize_value(&data, &length, &mOutHeight);
  deserialize_value(&data, &length, &mOutZ);
}

nvinfer1::IPluginV2DynamicExt *TRTBEVPoolV2::clone() const TRT_NOEXCEPT {
  TRTBEVPoolV2 *plugin = new TRTBEVPoolV2(mLayerName, mOutWidth, mOutHeight, mOutZ);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTBEVPoolV2::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  // input[0] == depth
  // input[1] == feat
  // input[2] == ranks_depth
  // input[3] == ranks_feat
  // input[4] == ranks_bev
  nvinfer1::DimsExprs ret; // B Z Y X C
  ret.nbDims = 5;
  ret.d[0] = exprBuilder.constant(1); //Todo support batch>1
  ret.d[1] = exprBuilder.constant(mOutZ); // Z
  ret.d[2] = exprBuilder.constant(mOutHeight); // Y
  ret.d[3] = exprBuilder.constant(mOutWidth); // X
  ret.d[4] = inputs[1].d[3]; // feat C
  return ret;
}

/**
 * @brief Check if the plugin supports the given format combination.
 *
 * This function verifies whether the plugin supports the format combination
 * for the given position in the input/output tensor descriptors.
 *
 * @param pos The position of the tensor descriptor in the input/output list.
 * @param ioDesc Pointer to the array of tensor descriptors.
 * @param nbInputs Number of input tensors.
 * @param nbOutputs Number of output tensors.
 * @return True if the format combination is supported, false otherwise.
 *
 * The function checks the following conditions:
 * - For positions 0, 1, and 7 (input[0], input[1], and output[0]):
 *   - The data type must be either kHALF, kINT8, or kFLOAT.
 *   - The tensor format must be kLINEAR.
 * - For all other positions (input[2] to input[6]):
 *   - The data type must be kINT32.
 *   - The tensor format must be kLINEAR.
 */
bool TRTBEVPoolV2::supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *ioDesc,
                 int32_t nbInputs, int32_t nbOutputs) TRT_NOEXCEPT {
  // input[0] == depth->kFLOAT or kHALF or kINT8
  // input[1] == feat->kFLOAT or kHALF or kINT8
  // input[2] == ranks_depth->kINT32
  // input[3] == ranks_feat->kINT32
  // input[4] == ranks_bev->kINT32
  // input[5] == interval_starts->kINT32
  // input[6] == interval_lengths->kINT32
  // output[0] == bev_feat->kFLOAT or kHALF or kINT8
  if (pos == 0 || pos == 1 || pos == 7) {
    // return (ioDesc[pos].type == nvinfer1::DataType::kHALF || 
    //         ioDesc[pos].type == nvinfer1::DataType::kINT8 ||
    //         ioDesc[pos].type == nvinfer1::DataType::kFLOAT) &&
    //    ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;

    return (ioDesc[pos].type == nvinfer1::DataType::kHALF ||
            ioDesc[pos].type == nvinfer1::DataType::kFLOAT) &&
      ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
       ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
}

void TRTBEVPoolV2::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int32_t nbInputs,
             const nvinfer1::DynamicPluginTensorDesc *outputs,
             int32_t nbOutputs) TRT_NOEXCEPT {
  // Validate input arguments

  ASSERT(nbInputs == 7);
  ASSERT(nbOutputs == 1);
}

size_t TRTBEVPoolV2::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs,
                const nvinfer1::PluginTensorDesc *outputs,
                int32_t nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int32_t TRTBEVPoolV2::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
            const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
            void *const *outputs, void *workSpace,
            cudaStream_t stream) TRT_NOEXCEPT {
  nvinfer1::Dims feat_dims = inputDesc[1].dims; // bnhwc
  nvinfer1::Dims interval_dims = inputDesc[5].dims; // n
  nvinfer1::Dims out_dims = outputDesc[0].dims; //bhwc
  std::cout << "inputDesc[0] type: " << (inputDesc[0].type == nvinfer1::DataType::kFLOAT ? "float" : "half") << std::endl;
  std::cout << "inputDesc[1] type: " << (inputDesc[1].type == nvinfer1::DataType::kFLOAT ? "float" : "half") << std::endl;
  // assert(inputDesc[0].type == inputDesc[1].type);
  auto data_type = inputDesc[0].type;
  int32_t num_points = out_dims.d[0]*out_dims.d[1]*out_dims.d[2]*out_dims.d[3] * out_dims.d[4];
  auto depth_type = inputDesc[0].type;
  auto feat_type = inputDesc[1].type;
  float scale_depth = depth_type == nvinfer1::DataType::kINT8 ? inputDesc[0].scale : 1.0;
  float scale_feat = feat_type == nvinfer1::DataType::kINT8 ? inputDesc[1].scale : 1.0;
  float scale_out = outputDesc[0].scale;

  if(nvinfer1::DataType::kFLOAT == depth_type)
  {
    bev_pool_v2_set_zero(num_points, (float *)outputs[0]);
  }
  else if(nvinfer1::DataType::kHALF == depth_type)
  {
    bev_pool_v2_set_zero(num_points, (__half *)outputs[0]);
  }
  else if(nvinfer1::DataType::kINT8 == depth_type)
  {
    bev_pool_v2_set_zero(num_points, (int8_t *)outputs[0]);
  }

  if(nvinfer1::DataType::kFLOAT == depth_type && nvinfer1::DataType::kFLOAT == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (float *)inputs[0], (float *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5],(int32_t *)inputs[6], (float *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kHALF == depth_type && nvinfer1::DataType::kHALF == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (__half *)inputs[0], (__half *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (__half *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kINT8 == depth_type && nvinfer1::DataType::kINT8 == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (int8_t *)inputs[0], (int8_t *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (int8_t *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kFLOAT == depth_type && nvinfer1::DataType::kHALF == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (float *)inputs[0], (__half *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (float *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kFLOAT == depth_type && nvinfer1::DataType::kINT8 == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (float *)inputs[0], (int8_t *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (float *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kHALF == depth_type && nvinfer1::DataType::kFLOAT == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (__half *)inputs[0], (float *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (__half *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kHALF == depth_type && nvinfer1::DataType::kINT8 == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (__half *)inputs[0], (int8_t *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (__half *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kINT8 == depth_type && nvinfer1::DataType::kFLOAT == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (int8_t *)inputs[0], (float *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (int8_t *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else if(nvinfer1::DataType::kINT8 == depth_type && nvinfer1::DataType::kHALF == feat_type){
    bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (int8_t *)inputs[0], (__half *)inputs[1],
      (int32_t *)inputs[2], (int32_t *)inputs[3], (int32_t *)inputs[4], (int32_t *)inputs[5], (int32_t *)inputs[6], (int8_t *)outputs[0],
      scale_depth, scale_feat, scale_out, stream);
  }else{
    std::cerr << "Unsupported data type" << std::endl;
  }
  return 0;
}

nvinfer1::DataType TRTBEVPoolV2::getOutputDataType(int32_t index,
                     const nvinfer1::DataType *inputTypes,
                     int32_t nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTBEVPoolV2::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTBEVPoolV2::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int32_t TRTBEVPoolV2::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTBEVPoolV2::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mOutWidth) + serialized_size(mOutHeight) + serialized_size(mOutZ);
}

void TRTBEVPoolV2::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mOutWidth);
  serialize_value(&buffer, mOutHeight);
  serialize_value(&buffer, mOutZ);
}

////////////////////// creator /////////////////////////////

TRTBEVPoolV2Creator::TRTBEVPoolV2Creator() {
  mPluginAttributes = std::vector<nvinfer1::PluginField>(
  {nvinfer1::PluginField("out_z"), nvinfer1::PluginField("out_height"), nvinfer1::PluginField("out_width")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTBEVPoolV2Creator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTBEVPoolV2Creator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::createPlugin(
  
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  int32_t outWidth = 352;
  int32_t outHeight = 96;
  int32_t outZ = 16;
  for (int32_t i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
  continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("out_height") == 0) {
  outHeight = static_cast<const int32_t *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("out_width") == 0) {
  outWidth = static_cast<const int32_t *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("out_z") == 0) {
  outZ = static_cast<const int32_t *>(fc->fields[i].data)[0];
    }
  }
  ASSERT(outHeight > 0);
  ASSERT(outWidth > 0);
  ASSERT(outZ > 0);
  TRTBEVPoolV2 *plugin = new TRTBEVPoolV2(name, outWidth, outHeight, outZ);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::deserializePlugin(const char *name,
                      const void *serialData,
                      size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TRTBEVPoolV2(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTBEVPoolV2Creator);
}  // namespace mmdeploy
