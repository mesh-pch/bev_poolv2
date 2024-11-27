#include <stdio.h>
#include <stdlib.h>
#include "trt_bev_pool_kernel.hpp"

template <typename T>
__device__ inline float convert_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float convert_to_float<__half>(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ inline T convert_from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline int8_t convert_from_float<int8_t>(float value) {
    return __float2int_rn(value);
}

template <>
__device__ inline __half convert_from_float<__half>(float value) {
    return __float2half(value);
}


/**
 * @brief Kernel function for BEV Pooling
 * 
 * @param c                Number of channels
 * @param n_intervals      Number of unique points
 * @param depth            Input depth, FloatTensor[b, n, d, h, w]
 * @param feat             Input feature, FloatTensor[b, n, h, w, c]
 * @param ranks_depth      Input index of depth, int32_tTensor[n_points]
 * @param ranks_feat       Input index of feature, int32_tTensor[n_points]
 * @param ranks_bev        Output index, int32_tTensor[n_points]
 * @param interval_lengths Starting position for pooled point, int32_tTensor[n_intervals]
 * @param interval_starts  Number of points in each pooled point, int32_tTensor[n_intervals]
 * @param out              Output features, FloatTensor[b, z, h, w, c]
 */

template <typename T1, typename T2, typename T3>
__global__ void bev_pool_v2_kernel(int32_t c, int32_t n_int32_tervals,
                                  const T1 *__restrict__ depth,
                                  const T2 *__restrict__ feat,
                                  const int32_t *__restrict__ ranks_depth,
                                  const int32_t *__restrict__ ranks_feat,
                                  const int32_t *__restrict__ ranks_bev,
                                  const int32_t *__restrict__ int32_terval_starts,
                                  const int32_t *__restrict__ int32_terval_lengths,
                                  T3* __restrict__ out,
                                  float scale_depth, float scale_feat, float scale_out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t index = idx / c;
  int32_t cur_c = idx % c;
  if (index >= n_int32_tervals) return;
  int32_t int32_terval_start = int32_terval_starts[index];
  int32_t int32_terval_length = int32_terval_lengths[index];
  float psum = 0;
  const T1* cur_depth;
  const T2* cur_feat;
  for(int32_t i = 0; i < int32_terval_length; i++){
    cur_depth = depth + ranks_depth[int32_terval_start+i];
    cur_feat = feat + ranks_feat[int32_terval_start+i] * c + cur_c;
    psum += convert_to_float(*cur_feat) * convert_to_float(*cur_depth) * scale_feat * scale_depth;
    // psum += (*cur_feat) * (*cur_depth);
  }

  const int32_t* cur_rank = ranks_bev + int32_terval_start;
  T3* cur_out = out + *cur_rank * c + cur_c;
  *cur_out = convert_from_float<T3>(psum / scale_out);
}


template <typename T>
__global__ void bev_pool_v2_set_zero_kernel(int32_t n_point32_ts, T* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_point32_ts) return;
  float* cur_out = out + idx;
  *cur_out = 0.0;
}

template <>
__global__ void  bev_pool_v2_set_zero_kernel<int8_t>(int32_t n_point32_ts, int8_t* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_point32_ts) return;
  int8_t* cur_out = out + idx;
  *cur_out = 0;
}

template <>
__global__ void  bev_pool_v2_set_zero_kernel<__half>(int32_t n_point32_ts, __half* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_point32_ts) return;
  __half* cur_out = out + idx;
  *cur_out = __float2half(0.0f);
}

template <typename T1, typename T2, typename T3>
void bev_pool_v2(int32_t c, int32_t n_int32_tervals, const T1* depth, const T2* feat, const int32_t* ranks_depth, const int32_t* ranks_feat,
  const int32_t* ranks_bev, const int32_t* int32_terval_starts, const int32_t* int32_terval_lengths, T3* out, float scale_depth, float scale_feat, float scale_out,
  cudaStream_t stream) {
  bev_pool_v2_kernel<<<(int32_t)ceil(((double)n_int32_tervals * c / 256)), 256, 0, stream>>>(
    c, n_int32_tervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, int32_terval_starts, int32_terval_lengths, out, scale_depth, scale_feat, scale_out
  );
}

template <typename T>
void bev_pool_v2_set_zero(int32_t n_point32_ts, T* out) {
  bev_pool_v2_set_zero_kernel<<<(int32_t)ceil(((double)n_point32_ts / 256)), 256>>>(n_point32_ts, out);
}

template void bev_pool_v2<float, float, float>(int32_t, int32_t, const float*, const float*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, float*, float, float, float, cudaStream_t);
template void bev_pool_v2<__half, __half, __half>(int32_t, int32_t, const __half*, const __half*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, __half*, float, float, float, cudaStream_t);
template void bev_pool_v2<int8_t, int8_t, int8_t>(int32_t, int32_t, const int8_t*, const int8_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int8_t*, float, float, float, cudaStream_t);

template void bev_pool_v2<float, __half, float>(int32_t, int32_t, const float*, const __half*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, float*, float, float, float, cudaStream_t);
template void bev_pool_v2<float, int8_t, float>(int32_t, int32_t, const float*, const int8_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, float*, float, float, float, cudaStream_t);
template void bev_pool_v2<__half, float, __half>(int32_t, int32_t, const __half*, const float*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, __half*, float, float, float, cudaStream_t);
template void bev_pool_v2<__half, int8_t, __half>(int32_t, int32_t, const __half*, const int8_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, __half*, float, float, float, cudaStream_t);
template void bev_pool_v2<int8_t, float, int8_t>(int32_t, int32_t, const int8_t*, const float*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int8_t*, float, float, float, cudaStream_t);
template void bev_pool_v2<int8_t, __half, int8_t>(int32_t, int32_t, const int8_t*, const __half*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int8_t*, float, float, float, cudaStream_t);

template void bev_pool_v2_set_zero<float>(int32_t, float*);
template void bev_pool_v2_set_zero<__half>(int32_t, __half*);
template void bev_pool_v2_set_zero<int8_t>(int32_t, int8_t*);