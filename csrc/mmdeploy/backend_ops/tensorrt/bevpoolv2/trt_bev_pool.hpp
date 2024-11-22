// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_BEV_POOL_HPP
#define TRT_BEV_POOL_HPP
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>
#include <cstdint>  // Include for int32_t

#include "trt_plugin_base.hpp"

namespace mmdeploy {

class TRTBEVPoolV2 : public TRTPluginBase {
 public:
    TRTBEVPoolV2(const std::string &name, int32_t outWidth, int32_t outHeight, int32_t outZ);

    TRTBEVPoolV2(const std::string name, const void *data, size_t length);

    TRTBEVPoolV2() = delete;

    ~TRTBEVPoolV2() TRT_NOEXCEPT override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override;

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs,
                                                                                    int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder)
            TRT_NOEXCEPT override;

    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *ioDesc, int32_t nbInputs,
                                                                 int32_t nbOutputs) TRT_NOEXCEPT override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc *out,
                                             int32_t nbOutputs) TRT_NOEXCEPT override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs,
                                                    const nvinfer1::PluginTensorDesc *outputs,
                                                    int32_t nbOutputs) const TRT_NOEXCEPT override;

    int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                            const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                            void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
                                                                             int32_t nbInputs) const TRT_NOEXCEPT override;

    // IPluginV2 Methods
    const char *getPluginType() const TRT_NOEXCEPT override;

    const char *getPluginVersion() const TRT_NOEXCEPT override;

    int32_t getNbOutputs() const TRT_NOEXCEPT override;

    size_t getSerializationSize() const TRT_NOEXCEPT override;

    void serialize(void *buffer) const TRT_NOEXCEPT override;

 private:
    int32_t mOutWidth;
    int32_t mOutHeight;
    int32_t mOutZ;
};

class TRTBEVPoolV2Creator : public TRTPluginCreatorBase {
 public:
    TRTBEVPoolV2Creator();

    ~TRTBEVPoolV2Creator() TRT_NOEXCEPT override = default;

    const char *getPluginName() const TRT_NOEXCEPT override;

    const char *getPluginVersion() const TRT_NOEXCEPT override;

    nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
            TRT_NOEXCEPT override;

    nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                        size_t serialLength) TRT_NOEXCEPT override;
};
}  // namespace mmdeploy
#endif  // TRT_GRID_SAMPLER_HPP
