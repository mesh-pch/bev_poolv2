#ifndef TRT_BEV_POOL_KERNEL_HPP
#define TRT_BEV_POOL_KERNEL_HPP
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common_cuda_helper.hpp"

// CUDA function declarations
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
  const int32_t* ranks_depth, const int32_t* ranks_feat, const int32_t* ranks_bev,
  const int32_t* interval_starts, const int32_t* interval_lengths, float* out, cudaStream_t stream);

void bev_pool_v2_half(int32_t c, int32_t n_intervals, const __half* depth, const __half* feat, const int32_t* ranks_depth,
  const int32_t* ranks_feat, const int32_t* ranks_bev, const int32_t* interval_starts, const int32_t* interval_lengths, __half* out,
  cudaStream_t stream);

void bev_pool_v2_int8(int32_t c, int32_t n_intervals, const int8_t* depth, const int8_t* feat, const int32_t* ranks_depth,
  const int32_t* ranks_feat, const int32_t* ranks_bev, const int32_t* interval_starts, const int32_t* interval_lengths, int8_t* out,
  float scale_depth, float scale_feat, float scale_out, cudaStream_t stream);  

void bev_pool_v2_set_zero(int32_t n_points, float* out);
void bev_pool_v2_set_zero_half(int32_t n_points, __half* out);
void bev_pool_v2_set_zero_int8(int32_t n_points, int8_t* out);
#endif // TRT_BEV_POOL_KERNEL_HPP