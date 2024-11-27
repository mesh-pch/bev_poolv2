#ifndef TRT_BEV_POOL_KERNEL_HPP
#define TRT_BEV_POOL_KERNEL_HPP
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common_cuda_helper.hpp"

// CUDA function declarations
template <typename T>
void bev_pool_v2_set_zero(int32_t n_point32_ts, T* out);

template <typename T1, typename T2, typename T3>
void bev_pool_v2(int32_t c, int32_t n_int32_tervals, const T1* depth, const T2* feat, const int32_t* ranks_depth, const int32_t* ranks_feat,
  const int32_t* ranks_bev, const int32_t* int32_terval_starts, const int32_t* int32_terval_lengths, T3* out, float scale_depth, float scale_feat, float scale_out,
  cudaStream_t stream);
#endif // TRT_BEV_POOL_KERNEL_HPP