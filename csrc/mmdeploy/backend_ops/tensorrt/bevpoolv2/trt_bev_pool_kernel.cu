#include <stdio.h>
#include <stdlib.h>
#include "trt_bev_pool_kernel.hpp"
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
__global__ void bev_pool_v2_kernel(int32_t c, int32_t n_intervals,
                                  const float *__restrict__ depth,
                                  const float *__restrict__ feat,
                                  const int32_t *__restrict__ ranks_depth,
                                  const int32_t *__restrict__ ranks_feat,
                                  const int32_t *__restrict__ ranks_bev,
                                  const int32_t *__restrict__ interval_starts,
                                  const int32_t *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t index = idx / c;
  int32_t cur_c = idx % c;
  if (index >= n_intervals) return;
  int32_t interval_start = interval_starts[index];
  int32_t interval_length = interval_lengths[index];
  float psum = 0;
  const float* cur_depth;
  const float* cur_feat;
  for(int32_t i = 0; i < interval_length; i++){
    cur_depth = depth + ranks_depth[interval_start+i];
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    psum += (*cur_feat) * (*cur_depth);
  }

  const int32_t* cur_rank = ranks_bev + interval_start;
  float* cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

__global__ void bev_pool_v2_set_zero_kernel(int32_t n_points, float* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  float* cur_out = out + idx;
  *cur_out = 0.0;
}

void bev_pool_v2(int32_t c, int32_t n_intervals, const float* depth, const float* feat, const int32_t* ranks_depth,
  const int32_t* ranks_feat, const int32_t* ranks_bev, const int32_t* interval_starts, const int32_t* interval_lengths, float* out,
  cudaStream_t stream) {
  bev_pool_v2_kernel<<<(int32_t)ceil(((double)n_intervals * c / 256)), 256, 0, stream>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}


void bev_pool_v2_set_zero(int32_t n_points, float* out) {
  bev_pool_v2_set_zero_kernel<<<(int32_t)ceil(((double)n_points / 256)), 256>>>(n_points, out);
}

/**
 * @brief Kernel function for BEV Pooling.
 * 
 * @param c Number of channels.
 * @param n_intervals Number of unique points.
 * @param depth Input depth, __half[b,n,d,h,w].
 * @param feat Input feature, __half[b,n,h,w,c].
 * @param ranks_depth Input index of depth, int32_t[n].
 * @param ranks_feat Input index of feature, int32_t[n].
 * @param ranks_bev Output index, int32_t[n].
 * @param interval_lengths Starting position for pooled point, int32_t[n_intervals].
 * @param interval_starts Number of points in each pooled point, int32_t[n_intervals].
 * @param out Output features, __half[b,d,h,w,c].
 */
__global__ void bev_pool_v2_kernel_half(int32_t c, int32_t n_intervals,
                                  const __half *__restrict__ depth,
                                  const __half *__restrict__ feat,
                                  const int32_t *__restrict__ ranks_depth,
                                  const int32_t *__restrict__ ranks_feat,
                                  const int32_t *__restrict__ ranks_bev,
                                  const int32_t *__restrict__ interval_starts,
                                  const int32_t *__restrict__ interval_lengths,
                                  __half* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t index = idx / c;
  int32_t cur_c = idx % c;
  if (index >= n_intervals) return;
  int32_t interval_start = interval_starts[index];
  int32_t interval_length = interval_lengths[index];
  float psum = 0;
  const __half* cur_depth;
  const __half* cur_feat;
  for(int32_t i = 0; i < interval_length; i++){
    cur_depth = depth + ranks_depth[interval_start+i];
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    psum += __half2float(*cur_feat) * __half2float(*cur_depth);
  }

  const int32_t* cur_rank = ranks_bev + interval_start;
  __half* cur_out = out + *cur_rank * c + cur_c;

  *cur_out = __float2half(psum);
}

__global__ void bev_pool_v2_set_zero_kernel_half(int32_t n_points, __half* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  __half* cur_out = out + idx;
  *cur_out = __float2half(0.0f);
}

void bev_pool_v2_half(int32_t c, int32_t n_intervals, const __half* depth, const __half* feat, const int32_t* ranks_depth,
  const int32_t* ranks_feat, const int32_t* ranks_bev, const int32_t* interval_starts, const int32_t* interval_lengths, __half* out,
  cudaStream_t stream) {

  bev_pool_v2_kernel_half<<<(int32_t)ceil(((double)n_intervals * c / 256)), 256, 0, stream>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out
  );
}

void bev_pool_v2_set_zero_half(int32_t n_points, __half* out) {
  bev_pool_v2_set_zero_kernel_half<<<(int32_t)ceil(((double)n_points / 256)), 256>>>(n_points, out);
}

/**
 * @brief Kernel function for BEV Pooling with int8 quantization.
 *
 * This kernel performs BEV (Bird's Eye View) pooling on quantized int8 input data.
 *
 * @param c                Number of channels.
 * @param n_intervals      Number of unique points.
 * @param depth            Input depth, int8_t[b, n, d, h, w].
 * @param feat             Input feature, int8_t[b, n, h, w, c].
 * @param ranks_depth      Input index of depth, int32_t[n].
 * @param ranks_feat       Input index of feature, int32_t[n].
 * @param ranks_bev        Output index, int32_t[n].
 * @param interval_starts  Starting position for pooled point, int32_t[n_intervals].
 * @param interval_lengths Number of points in each pooled point, int32_t[n_intervals].
 * @param out              Output features, int8_t[b, d, h, w, c].
 * @param scale_depth      Scaling factor for depth data.
 * @param scale_feat       Scaling factor for feature data.
 * @param scale_out        Scaling factor for output data.
 */
__global__ void bev_pool_v2_kernel_int8_quantized(int32_t c, int32_t n_intervals,
                                  const int8_t *__restrict__ depth,
                                  const int8_t *__restrict__ feat,
                                  const int32_t *__restrict__ ranks_depth,
                                  const int32_t *__restrict__ ranks_feat,
                                  const int32_t *__restrict__ ranks_bev,
                                  const int32_t *__restrict__ interval_starts,
                                  const int32_t *__restrict__ interval_lengths,
                                  int8_t* __restrict__ out,
                                  float scale_depth, float scale_feat, float scale_out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t index = idx / c;
  int32_t cur_c = idx % c;
  if (index >= n_intervals) return;
  int32_t interval_start = interval_starts[index];
  int32_t interval_length = interval_lengths[index];
  float psum = 0;
  const int8_t* cur_depth;
  const int8_t* cur_feat;
  for(int32_t i = 0; i < interval_length; i++){
    cur_depth = depth + ranks_depth[interval_start+i];
    cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
    psum += (*cur_feat) * (*cur_depth) * scale_feat * scale_depth;
  }

  const int32_t* cur_rank = ranks_bev + interval_start;
  int8_t* cur_out = out + *cur_rank * c + cur_c;
  *cur_out = __float2int_rn(psum / scale_out);
}

__global__ void bev_pool_v2_set_zero_kernel_int8(int32_t n_points, int8_t* __restrict__ out) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_points) return;
  int8_t* cur_out = out + idx;
  *cur_out = 0;
}

void bev_pool_v2_set_zero_int8(int32_t n_points, int8_t* out) {
  bev_pool_v2_set_zero_kernel_int8<<<(int32_t)ceil(((double)n_points / 256)), 256>>>(n_points, out);
}
void bev_pool_v2_int8(int32_t c, int32_t n_intervals, const int8_t* depth, const int8_t* feat, const int32_t* ranks_depth,
  const int32_t* ranks_feat, const int32_t* ranks_bev, const int32_t* interval_starts, const int32_t* interval_lengths, int8_t* out,
  float scale_depth, float scale_feat, float scale_out, cudaStream_t stream) {

  bev_pool_v2_kernel_int8_quantized<<<(int32_t)ceil(((double)n_intervals * c / 256)), 256, 0, stream>>>(
    c, n_intervals, depth, feat, ranks_depth, ranks_feat,
    ranks_bev, interval_starts, interval_lengths, out, scale_depth, scale_feat, scale_out
  );
}

