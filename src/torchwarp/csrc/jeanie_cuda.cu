// Fused JEANIE dynamic program (forward + analytic backward).
//
// Global layout: [B, T*U, K] with K = K1*K2 viewpoints innermost and the
// (t, u) cells in diagonal-major order (diagonal by diagonal, row order
// within a diagonal). A whole anti-diagonal (all rows, all viewpoints) is
// therefore one contiguous slab, so every global access is coalesced, and the
// next slab is prefetched into registers while the current one is computed.
//
// One thread block handles one pair; thread `tid` owns slab entries
// tid + r * blockDim. The last three diagonals of the accumulator live in
// shared memory, indexed by (row t, viewpoint n).
//
// Forward (Algorithm 1 of the JEANIE paper, extended to two viewpoint axes):
//   R[n,0,0] = C[n,0,0]
//   R[n,t,u] = C[n,t,u] + SoftMin_gamma{ R[n',t-j,u-k] :
//                |n1'-n1| <= s1, |n2'-n2| <= s2, (j,k) in {(1,0),(0,1),(1,1)} }
//   out      = SoftMin_gamma{ R[n,T-1,U-1] }
// soft-DTW is the special case K1 = K2 = 1.
//
// Backward: with E_s = R_s - C_s (the SoftMin value at successor s),
//   Rbar_k = sum_s exp((E_s - R_k) / gamma) * Rbar_s,   dL/dC = Rbar,
// seeded with Rbar[n,T-1,U-1] = grad_out * softmax_n(-R[n,T-1,U-1] / gamma),
// plus dL/dR[n,t,u] when the accumulator itself feeds the loss.
//
// Two kernel families share that core:
//   * cost kernels take a precomputed cost C;
//   * fused kernels take G = <q[n,t], s[u]> plus squared norms and build the
//     (squared) Euclidean cost in registers, returning dG, dq2, ds2.
//
// Portable constructs only (shared memory, __syncthreads, atomicAdd on
// shared memory): builds under ROCm via PyTorch's automatic hipify.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <vector>

namespace {

constexpr int kMaxThreads = 256;
constexpr size_t kMaxStaticSmem = 48 * 1024;

enum Metric : int { kSquared = 0, kEuclidean = 1 };

template <typename T>
__device__ __forceinline__ T dmin(T a, T b) { return a < b ? a : b; }

// exp/log used by the DP. With -DDTW_FAST_MATH, float32 uses the hardware
// approximations __expf/__logf (a few ulp); float64 always uses exp/log.
template <typename T>
__device__ __forceinline__ T dexp(T x) { return exp(x); }
template <typename T>
__device__ __forceinline__ T dlog(T x) { return log(x); }
#ifdef DTW_FAST_MATH
template <>
__device__ __forceinline__ float dexp<float>(float x) { return __expf(x); }
template <>
__device__ __forceinline__ float dlog<float>(float x) { return __logf(x); }
#endif

__device__ __forceinline__ int diag_lo(int d, int U) { return max(0, d - (U - 1)); }
__device__ __forceinline__ int diag_hi(int d, int T) { return min(T - 1, d); }
__device__ __forceinline__ int diag_len(int d, int T, int U) {
  return diag_hi(d, T) - diag_lo(d, U) + 1;
}

struct Grid {
  int K1, K2, K, T, U, s1, s2;
};

// SoftMin over the (2 s1 + 1)(2 s2 + 1) x 3 predecessors of (n, t, u).
// R1 / R2 are the accumulators of diagonals d-1 / d-2, indexed [t * K + n].
template <typename scalar_t>
__device__ __forceinline__ scalar_t pred_softmin(
    const scalar_t* R1, const scalar_t* R2, const Grid& g,
    int t, int u, int n, scalar_t gamma, scalar_t inv_g) {
  const int n1 = n / g.K2, n2 = n % g.K2;
  const int a1lo = max(0, n1 - g.s1), a1hi = min(g.K1 - 1, n1 + g.s1);
  const int a2lo = max(0, n2 - g.s2), a2hi = min(g.K2 - 1, n2 + g.s2);
  scalar_t m = (scalar_t)INFINITY;
  for (int a1 = a1lo; a1 <= a1hi; ++a1) {
    for (int a2 = a2lo; a2 <= a2hi; ++a2) {
      const int pn = a1 * g.K2 + a2;
      if (t > 0) m = dmin(m, R1[(t - 1) * g.K + pn]);
      if (u > 0) m = dmin(m, R1[t * g.K + pn]);
      if (t > 0 && u > 0) m = dmin(m, R2[(t - 1) * g.K + pn]);
    }
  }
  scalar_t s = 0;
  for (int a1 = a1lo; a1 <= a1hi; ++a1) {
    for (int a2 = a2lo; a2 <= a2hi; ++a2) {
      const int pn = a1 * g.K2 + a2;
      if (t > 0) s += dexp((m - R1[(t - 1) * g.K + pn]) * inv_g);
      if (u > 0) s += dexp((m - R1[t * g.K + pn]) * inv_g);
      if (t > 0 && u > 0) s += dexp((m - R2[(t - 1) * g.K + pn]) * inv_g);
    }
  }
  return m - gamma * dlog(s);
}

// Adjoint of R[n,t,u] collected from its successors on diagonals d+1 / d+2.
template <typename scalar_t>
__device__ __forceinline__ scalar_t succ_adjoint(
    const scalar_t* Gb, const scalar_t* Eb, int c1, int c2, const Grid& g,
    int t, int u, int n, scalar_t rk, scalar_t inv_g) {
  const int n1 = n / g.K2, n2 = n % g.K2;
  const int a1lo = max(0, n1 - g.s1), a1hi = min(g.K1 - 1, n1 + g.s1);
  const int a2lo = max(0, n2 - g.s2), a2hi = min(g.K2 - 1, n2 + g.s2);
  scalar_t acc = 0;
  for (int a1 = a1lo; a1 <= a1hi; ++a1) {
    for (int a2 = a2lo; a2 <= a2hi; ++a2) {
      const int sn = a1 * g.K2 + a2;
      if (t + 1 < g.T) {  // (t+1, u) on diagonal d+1
        const int s = c1 + (t + 1) * g.K + sn;
        acc += dexp((Eb[s] - rk) * inv_g) * Gb[s];
      }
      if (u + 1 < g.U) {  // (t, u+1) on diagonal d+1
        const int s = c1 + t * g.K + sn;
        acc += dexp((Eb[s] - rk) * inv_g) * Gb[s];
      }
      if (t + 1 < g.T && u + 1 < g.U) {  // (t+1, u+1) on diagonal d+2
        const int s = c2 + (t + 1) * g.K + sn;
        acc += dexp((Eb[s] - rk) * inv_g) * Gb[s];
      }
    }
  }
  return acc;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t final_softmin(const scalar_t* last, int K, scalar_t gamma) {
  const scalar_t inv_g = (scalar_t)1 / gamma;
  scalar_t m = (scalar_t)INFINITY;
  for (int n = 0; n < K; ++n) m = dmin(m, last[n]);
  scalar_t s = 0;
  for (int n = 0; n < K; ++n) s += dexp((m - last[n]) * inv_g);
  return m - gamma * dlog(s);
}

// ------------------------------- cost kernels ------------------------------

template <typename scalar_t, int ITEMS>
__global__ void jeanie_forward_kernel(
    const scalar_t* __restrict__ C,
    scalar_t* __restrict__ R,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ scratch,
    const Grid g,
    const scalar_t gamma) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int K = g.K, T = g.T, U = g.U;
  const int b = blockIdx.x;
  const long long off_b = (long long)b * T * U * K;
  C += off_b; R += off_b;
  const int slab = T * K;
  scalar_t* buf = scratch != nullptr
      ? scratch + (long long)b * 3 * slab
      : reinterpret_cast<scalar_t*>(smem_raw);

  const scalar_t inv_g = (scalar_t)1 / gamma;
  const int D = T + U - 1;

  scalar_t c_nx[ITEMS];
#pragma unroll
  for (int r = 0; r < ITEMS; ++r) {
    const int idx = threadIdx.x + r * blockDim.x;
    if (idx < K) c_nx[r] = C[idx];
  }

  long long off = 0;  // start of diagonal d (in elements)
  for (int d = 0; d < D; ++d) {
    const int lo = diag_lo(d, U);
    const int cells = diag_len(d, T, U) * K;

    scalar_t c_cur[ITEMS];
#pragma unroll
    for (int r = 0; r < ITEMS; ++r) c_cur[r] = c_nx[r];

    if (d + 1 < D) {  // prefetch diagonal d+1
      const int cells1 = diag_len(d + 1, T, U) * K;
#pragma unroll
      for (int r = 0; r < ITEMS; ++r) {
        const int idx = threadIdx.x + r * blockDim.x;
        if (idx < cells1) c_nx[r] = C[off + cells + idx];
      }
    }

    scalar_t* Rc = buf + (d % 3) * slab;
    const scalar_t* R1 = buf + ((d + 2) % 3) * slab;  // diagonal d-1
    const scalar_t* R2 = buf + ((d + 1) % 3) * slab;  // diagonal d-2

#pragma unroll
    for (int r = 0; r < ITEMS; ++r) {
      const int idx = threadIdx.x + r * blockDim.x;
      if (idx >= cells) continue;
      const int t = lo + idx / K;
      const int n = idx % K;
      const int u = d - t;
      scalar_t rv = c_cur[r];
      if (d > 0) rv += pred_softmin(R1, R2, g, t, u, n, gamma, inv_g);
      Rc[t * K + n] = rv;
      R[off + idx] = rv;
    }
    __syncthreads();
    off += cells;
  }

  if (threadIdx.x == 0) {
    out[b] = final_softmin(buf + ((D - 1) % 3) * slab + (T - 1) * K, K, gamma);
  }
}

template <typename scalar_t, int ITEMS>
__global__ void jeanie_backward_kernel(
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ R,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ grad_R,  // optional dL/dR (same layout), may be null
    scalar_t* __restrict__ dC,
    scalar_t* __restrict__ scratch,
    const Grid g,
    const scalar_t gamma) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int K = g.K, T = g.T, U = g.U;
  const int b = blockIdx.x;
  const long long off_b = (long long)b * T * U * K;
  C += off_b; R += off_b; dC += off_b;
  if (grad_R != nullptr) grad_R += off_b;
  const int slab = T * K;
  scalar_t* buf = scratch != nullptr
      ? scratch + (long long)b * 6 * slab
      : reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* Gb = buf;             // Rbar   [3][T*K]
  scalar_t* Eb = buf + 3 * slab;  // R - C  [3][T*K]

  const scalar_t inv_g = (scalar_t)1 / gamma;
  const scalar_t gout = grad_out[b];
  const int D = T + U - 1;

  // The last diagonal is the single cell (T-1, U-1): K contiguous values.
  long long off = (long long)(T * U - 1) * K;
  const scalar_t fin = final_softmin(R + off, K, gamma);

  scalar_t r_nx[ITEMS], c_nx[ITEMS];
#pragma unroll
  for (int r = 0; r < ITEMS; ++r) {
    const int idx = threadIdx.x + r * blockDim.x;
    if (idx < K) { r_nx[r] = R[off + idx]; c_nx[r] = C[off + idx]; }
  }

  for (int d = D - 1; d >= 0; --d) {
    const int lo = diag_lo(d, U);
    const int cells = diag_len(d, T, U) * K;

    scalar_t r_cur[ITEMS], c_cur[ITEMS];
#pragma unroll
    for (int r = 0; r < ITEMS; ++r) { r_cur[r] = r_nx[r]; c_cur[r] = c_nx[r]; }

    long long off1 = 0;
    if (d > 0) {  // prefetch diagonal d-1
      const int cells1 = diag_len(d - 1, T, U) * K;
      off1 = off - cells1;
#pragma unroll
      for (int r = 0; r < ITEMS; ++r) {
        const int idx = threadIdx.x + r * blockDim.x;
        if (idx < cells1) { r_nx[r] = R[off1 + idx]; c_nx[r] = C[off1 + idx]; }
      }
    }

    const int c0 = (d % 3) * slab;
    const int c1 = ((d + 1) % 3) * slab;  // diagonal d+1
    const int c2 = ((d + 2) % 3) * slab;  // diagonal d+2

#pragma unroll
    for (int r = 0; r < ITEMS; ++r) {
      const int idx = threadIdx.x + r * blockDim.x;
      if (idx >= cells) continue;
      const int t = lo + idx / K;
      const int n = idx % K;
      const int u = d - t;
      const scalar_t rk = r_cur[r];
      scalar_t acc = (d == D - 1)
          ? gout * dexp((fin - rk) * inv_g)
          : succ_adjoint(Gb, Eb, c1, c2, g, t, u, n, rk, inv_g);
      if (grad_R != nullptr) acc += grad_R[off + idx];
      Gb[c0 + t * K + n] = acc;
      Eb[c0 + t * K + n] = rk - c_cur[r];
      dC[off + idx] = acc;
    }
    __syncthreads();
    off = off1;
  }
}

// ------------------------------ fused kernels ------------------------------
// G[b, cell, n] = <q[n,t], s[u]> (diagonal-major), q2[b, t, n] = ||q[n,t]||^2,
// s2[b, u] = ||s[u]||^2.  sq = max(q2 + s2 - 2 G, 0); cost = sq or sqrt(sq).
// The gradient follows torch.clamp_min (zero at sq = 0), and the Euclidean
// branch uses d sqrt(sq) = 0 at sq = 0, matching torchwarp.euclidean_cost.

template <typename scalar_t>
__device__ __forceinline__ scalar_t fused_cost(scalar_t sq_raw, int metric) {
  const scalar_t sq = sq_raw > (scalar_t)0 ? sq_raw : (scalar_t)0;
  return metric == kEuclidean ? sqrt(sq) : sq;
}

template <typename scalar_t, int ITEMS>
__global__ void jeanie_fused_forward_kernel(
    const scalar_t* __restrict__ G,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ s2,
    scalar_t* __restrict__ R,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ scratch,
    const Grid g,
    const scalar_t gamma,
    const int metric) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int K = g.K, T = g.T, U = g.U;
  const int b = blockIdx.x;
  const long long off_b = (long long)b * T * U * K;
  G += off_b; R += off_b;
  q2 += (long long)b * T * K;
  s2 += (long long)b * U;
  const int slab = T * K;
  scalar_t* buf = scratch != nullptr
      ? scratch + (long long)b * (3 * slab + U)
      : reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* ss = buf + 3 * slab;  // s2 [U]

  for (int k = threadIdx.x; k < U; k += blockDim.x) ss[k] = s2[k];

  const scalar_t inv_g = (scalar_t)1 / gamma;
  const int D = T + U - 1;

  scalar_t g_nx[ITEMS], q_nx[ITEMS];
#pragma unroll
  for (int r = 0; r < ITEMS; ++r) {
    const int idx = threadIdx.x + r * blockDim.x;
    if (idx < K) { g_nx[r] = G[idx]; q_nx[r] = q2[idx]; }
  }
  __syncthreads();

  long long off = 0;
  for (int d = 0; d < D; ++d) {
    const int lo = diag_lo(d, U);
    const int cells = diag_len(d, T, U) * K;

    scalar_t g_cur[ITEMS], q_cur[ITEMS];
#pragma unroll
    for (int r = 0; r < ITEMS; ++r) { g_cur[r] = g_nx[r]; q_cur[r] = q_nx[r]; }

    if (d + 1 < D) {  // prefetch diagonal d+1
      const int lo1 = diag_lo(d + 1, U);
      const int cells1 = diag_len(d + 1, T, U) * K;
#pragma unroll
      for (int r = 0; r < ITEMS; ++r) {
        const int idx = threadIdx.x + r * blockDim.x;
        if (idx < cells1) {
          g_nx[r] = G[off + cells + idx];
          q_nx[r] = q2[lo1 * K + idx];  // (t, n) = (lo1 + idx / K, idx % K)
        }
      }
    }

    scalar_t* Rc = buf + (d % 3) * slab;
    const scalar_t* R1 = buf + ((d + 2) % 3) * slab;
    const scalar_t* R2 = buf + ((d + 1) % 3) * slab;

#pragma unroll
    for (int r = 0; r < ITEMS; ++r) {
      const int idx = threadIdx.x + r * blockDim.x;
      if (idx >= cells) continue;
      const int t = lo + idx / K;
      const int n = idx % K;
      const int u = d - t;
      scalar_t rv = fused_cost(q_cur[r] + ss[u] - (scalar_t)2 * g_cur[r], metric);
      if (d > 0) rv += pred_softmin(R1, R2, g, t, u, n, gamma, inv_g);
      Rc[t * K + n] = rv;
      R[off + idx] = rv;
    }
    __syncthreads();
    off += cells;
  }

  if (threadIdx.x == 0) {
    out[b] = final_softmin(buf + ((D - 1) % 3) * slab + (T - 1) * K, K, gamma);
  }
}

template <typename scalar_t, int ITEMS>
__global__ void jeanie_fused_backward_kernel(
    const scalar_t* __restrict__ G,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ s2,
    const scalar_t* __restrict__ R,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ grad_R,  // optional dL/dR (same layout), may be null
    scalar_t* __restrict__ dG,
    scalar_t* __restrict__ dq2,
    scalar_t* __restrict__ ds2,
    scalar_t* __restrict__ scratch,
    const Grid g,
    const scalar_t gamma,
    const int metric) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int K = g.K, T = g.T, U = g.U;
  const int b = blockIdx.x;
  const long long off_b = (long long)b * T * U * K;
  G += off_b; R += off_b; dG += off_b;
  if (grad_R != nullptr) grad_R += off_b;
  q2 += (long long)b * T * K; dq2 += (long long)b * T * K;
  s2 += (long long)b * U; ds2 += (long long)b * U;
  const int slab = T * K;
  scalar_t* buf = scratch != nullptr
      ? scratch + (long long)b * (7 * slab + 2 * U)
      : reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* Gb = buf;              // Rbar   [3][T*K]
  scalar_t* Eb = buf + 3 * slab;   // R - C  [3][T*K]
  scalar_t* dqs = buf + 6 * slab;  // dq2    [T*K]
  scalar_t* ss = dqs + slab;       // s2     [U]
  scalar_t* dss = ss + U;          // ds2    [U]

  for (int k = threadIdx.x; k < slab; k += blockDim.x) dqs[k] = 0;
  for (int k = threadIdx.x; k < U; k += blockDim.x) { ss[k] = s2[k]; dss[k] = 0; }

  const scalar_t inv_g = (scalar_t)1 / gamma;
  const scalar_t gout = grad_out[b];
  const int D = T + U - 1;

  long long off = (long long)(T * U - 1) * K;
  const scalar_t fin = final_softmin(R + off, K, gamma);

  scalar_t r_nx[ITEMS], g_nx[ITEMS], q_nx[ITEMS];
#pragma unroll
  for (int r = 0; r < ITEMS; ++r) {
    const int idx = threadIdx.x + r * blockDim.x;
    if (idx < K) {
      r_nx[r] = R[off + idx]; g_nx[r] = G[off + idx]; q_nx[r] = q2[(T - 1) * K + idx];
    }
  }
  __syncthreads();

  for (int d = D - 1; d >= 0; --d) {
    const int lo = diag_lo(d, U);
    const int cells = diag_len(d, T, U) * K;

    scalar_t r_cur[ITEMS], g_cur[ITEMS], q_cur[ITEMS];
#pragma unroll
    for (int r = 0; r < ITEMS; ++r) { r_cur[r] = r_nx[r]; g_cur[r] = g_nx[r]; q_cur[r] = q_nx[r]; }

    long long off1 = 0;
    if (d > 0) {  // prefetch diagonal d-1
      const int lo1 = diag_lo(d - 1, U);
      const int cells1 = diag_len(d - 1, T, U) * K;
      off1 = off - cells1;
#pragma unroll
      for (int r = 0; r < ITEMS; ++r) {
        const int idx = threadIdx.x + r * blockDim.x;
        if (idx < cells1) {
          r_nx[r] = R[off1 + idx]; g_nx[r] = G[off1 + idx]; q_nx[r] = q2[lo1 * K + idx];
        }
      }
    }

    const int c0 = (d % 3) * slab;
    const int c1 = ((d + 1) % 3) * slab;
    const int c2 = ((d + 2) % 3) * slab;

#pragma unroll
    for (int r = 0; r < ITEMS; ++r) {
      const int idx = threadIdx.x + r * blockDim.x;
      if (idx >= cells) continue;
      const int t = lo + idx / K;
      const int n = idx % K;
      const int u = d - t;
      const scalar_t rk = r_cur[r];
      const scalar_t sq_raw = q_cur[r] + ss[u] - (scalar_t)2 * g_cur[r];
      const scalar_t c = fused_cost(sq_raw, metric);
      scalar_t acc = (d == D - 1)
          ? gout * dexp((fin - rk) * inv_g)
          : succ_adjoint(Gb, Eb, c1, c2, g, t, u, n, rk, inv_g);
      if (grad_R != nullptr) acc += grad_R[off + idx];
      Gb[c0 + t * K + n] = acc;
      Eb[c0 + t * K + n] = rk - c;

      scalar_t dsq = 0;
      if (sq_raw > (scalar_t)0) {
        dsq = metric == kEuclidean ? acc * (scalar_t)0.5 / c : acc;
      }
      dG[off + idx] = (scalar_t)-2 * dsq;
      dqs[t * K + n] += dsq;      // each (t, n) appears once per diagonal
      atomicAdd(&dss[u], dsq);    // K viewpoints share u on a diagonal
    }
    __syncthreads();
    off = off1;
  }

  for (int k = threadIdx.x; k < slab; k += blockDim.x) dq2[k] = dqs[k];
  for (int k = threadIdx.x; k < U; k += blockDim.x) ds2[k] = dss[k];
}

// ------------------------------- host helpers ------------------------------

struct LaunchShape {
  int threads;
  int items;
};

LaunchShape pick_shape(int cells) {
  int threads = ((cells + 31) / 32) * 32;
  threads = threads < 32 ? 32 : (threads > kMaxThreads ? kMaxThreads : threads);
  int items = (cells + threads - 1) / threads;
  int p = 1;
  while (p < items) p *= 2;
  TORCH_CHECK(p <= 64, "min(T, U) * K too large for the CUDA kernel (<= ",
              64 * kMaxThreads, ")");
  return {threads, p};
}

#define JEANIE_DISPATCH_ITEMS(VAR, ...)                            \
  switch (VAR) {                                                   \
    case 1: { constexpr int ITEMS = 1; __VA_ARGS__(); break; }     \
    case 2: { constexpr int ITEMS = 2; __VA_ARGS__(); break; }     \
    case 4: { constexpr int ITEMS = 4; __VA_ARGS__(); break; }     \
    case 8: { constexpr int ITEMS = 8; __VA_ARGS__(); break; }     \
    case 16: { constexpr int ITEMS = 16; __VA_ARGS__(); break; }   \
    case 32: { constexpr int ITEMS = 32; __VA_ARGS__(); break; }   \
    default: { constexpr int ITEMS = 64; __VA_ARGS__(); break; }   \
  }

Grid make_grid(int64_t K1, int64_t K2, int64_t T, int64_t U, int64_t s1, int64_t s2) {
  return Grid{(int)K1, (int)K2, (int)(K1 * K2), (int)T, (int)U, (int)s1, (int)s2};
}

// Per-block buffer of `elems` scalars: shared memory when it fits under the
// default limit, otherwise a global scratch tensor.
struct Buffer {
  torch::Tensor scratch;
  size_t smem;
};

Buffer make_buffer(const torch::Tensor& like, int64_t B, int64_t elems) {
  const size_t bytes = (size_t)elems * like.element_size();
  if (bytes <= kMaxStaticSmem) return {torch::Tensor(), bytes};
  return {torch::empty({B * elems}, like.options()), 0};
}

template <typename scalar_t>
scalar_t* ptr_or_null(const torch::Tensor& t) {
  return t.defined() ? t.data_ptr<scalar_t>() : nullptr;
}

void check_cells(const torch::Tensor& t, const char* name, int64_t T, int64_t U, int64_t K) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
  TORCH_CHECK(t.dim() == 3 && t.size(1) == T * U && t.size(2) == K,
              name, " must have shape [B, T*U, K] (diagonal-major)");
}

}  // namespace

std::vector<torch::Tensor> jeanie_forward(
    torch::Tensor C, int64_t K1, int64_t K2, int64_t T, int64_t U,
    int64_t s1, int64_t s2, double gamma) {
  check_cells(C, "cost", T, U, K1 * K2);
  const c10::cuda::OptionalCUDAGuard guard(C.device());
  const int B = C.size(0);
  auto R = torch::empty_like(C);
  auto out = torch::empty({B}, C.options());
  if (B == 0) return {out, R};

  const Grid g = make_grid(K1, K2, T, U, s1, s2);
  const Buffer buf = make_buffer(C, B, 3 * T * g.K);
  const LaunchShape shape = pick_shape((int)(std::min(T, U) * g.K));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(C.scalar_type(), "jeanie_forward", [&] {
    JEANIE_DISPATCH_ITEMS(shape.items, [&] {
      jeanie_forward_kernel<scalar_t, ITEMS><<<B, shape.threads, buf.smem, stream>>>(
          C.data_ptr<scalar_t>(), R.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
          ptr_or_null<scalar_t>(buf.scratch), g, (scalar_t)gamma);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, R};
}

torch::Tensor jeanie_backward(
    torch::Tensor C, torch::Tensor R, torch::Tensor grad_out,
    c10::optional<torch::Tensor> grad_R,
    int64_t K1, int64_t K2, int64_t T, int64_t U,
    int64_t s1, int64_t s2, double gamma) {
  check_cells(C, "cost", T, U, K1 * K2);
  check_cells(R, "R", T, U, K1 * K2);
  if (grad_R.has_value()) check_cells(*grad_R, "grad_R", T, U, K1 * K2);
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  const c10::cuda::OptionalCUDAGuard guard(C.device());
  const int B = C.size(0);
  auto dC = torch::empty_like(C);
  if (B == 0) return dC;

  const Grid g = make_grid(K1, K2, T, U, s1, s2);
  const Buffer buf = make_buffer(C, B, 6 * T * g.K);
  const LaunchShape shape = pick_shape((int)(std::min(T, U) * g.K));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(C.scalar_type(), "jeanie_backward", [&] {
    JEANIE_DISPATCH_ITEMS(shape.items, [&] {
      jeanie_backward_kernel<scalar_t, ITEMS><<<B, shape.threads, buf.smem, stream>>>(
          C.data_ptr<scalar_t>(), R.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
          grad_R.has_value() ? grad_R->data_ptr<scalar_t>() : nullptr,
          dC.data_ptr<scalar_t>(), ptr_or_null<scalar_t>(buf.scratch), g, (scalar_t)gamma);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dC;
}

std::vector<torch::Tensor> jeanie_fused_forward(
    torch::Tensor G, torch::Tensor q2, torch::Tensor s2,
    int64_t K1, int64_t K2, int64_t T, int64_t U,
    int64_t s1, int64_t s2_shift, double gamma, int64_t metric) {
  check_cells(G, "G", T, U, K1 * K2);
  TORCH_CHECK(q2.is_contiguous() && s2.is_contiguous(), "norms must be contiguous");
  const c10::cuda::OptionalCUDAGuard guard(G.device());
  const int B = G.size(0);
  auto R = torch::empty_like(G);
  auto out = torch::empty({B}, G.options());
  if (B == 0) return {out, R};

  const Grid g = make_grid(K1, K2, T, U, s1, s2_shift);
  const Buffer buf = make_buffer(G, B, 3 * T * g.K + U);
  const LaunchShape shape = pick_shape((int)(std::min(T, U) * g.K));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(G.scalar_type(), "jeanie_fused_forward", [&] {
    JEANIE_DISPATCH_ITEMS(shape.items, [&] {
      jeanie_fused_forward_kernel<scalar_t, ITEMS><<<B, shape.threads, buf.smem, stream>>>(
          G.data_ptr<scalar_t>(), q2.data_ptr<scalar_t>(), s2.data_ptr<scalar_t>(),
          R.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
          ptr_or_null<scalar_t>(buf.scratch), g, (scalar_t)gamma, (int)metric);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, R};
}

std::vector<torch::Tensor> jeanie_fused_backward(
    torch::Tensor G, torch::Tensor q2, torch::Tensor s2, torch::Tensor R,
    torch::Tensor grad_out, c10::optional<torch::Tensor> grad_R,
    int64_t K1, int64_t K2, int64_t T, int64_t U,
    int64_t s1, int64_t s2_shift, double gamma, int64_t metric) {
  check_cells(G, "G", T, U, K1 * K2);
  check_cells(R, "R", T, U, K1 * K2);
  if (grad_R.has_value()) check_cells(*grad_R, "grad_R", T, U, K1 * K2);
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  const c10::cuda::OptionalCUDAGuard guard(G.device());
  const int B = G.size(0);
  auto dG = torch::empty_like(G);
  auto dq2 = torch::empty_like(q2);
  auto ds2 = torch::empty_like(s2);
  if (B == 0) return {dG, dq2, ds2};

  const Grid g = make_grid(K1, K2, T, U, s1, s2_shift);
  const Buffer buf = make_buffer(G, B, 7 * T * g.K + 2 * U);
  const LaunchShape shape = pick_shape((int)(std::min(T, U) * g.K));
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(G.scalar_type(), "jeanie_fused_backward", [&] {
    JEANIE_DISPATCH_ITEMS(shape.items, [&] {
      jeanie_fused_backward_kernel<scalar_t, ITEMS><<<B, shape.threads, buf.smem, stream>>>(
          G.data_ptr<scalar_t>(), q2.data_ptr<scalar_t>(), s2.data_ptr<scalar_t>(),
          R.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
          grad_R.has_value() ? grad_R->data_ptr<scalar_t>() : nullptr,
          dG.data_ptr<scalar_t>(), dq2.data_ptr<scalar_t>(), ds2.data_ptr<scalar_t>(),
          ptr_or_null<scalar_t>(buf.scratch), g, (scalar_t)gamma, (int)metric);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dG, dq2, ds2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &jeanie_forward, "JEANIE forward on a cost tensor (CUDA/HIP)");
  m.def("backward", &jeanie_backward, "JEANIE backward on a cost tensor (CUDA/HIP)");
  m.def("fused_forward", &jeanie_fused_forward, "JEANIE forward from features (CUDA/HIP)");
  m.def("fused_backward", &jeanie_fused_backward, "JEANIE backward from features (CUDA/HIP)");
}
