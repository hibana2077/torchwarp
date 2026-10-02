// Fused uncertainty-DTW dynamic program (forward + analytic backward).
//
// One thread block handles one sequence pair. Threads sweep the cost matrix
// anti-diagonal by anti-diagonal; thread `tid` owns rows tid + r * blockDim.
// The last three diagonals live in shared memory, indexed by row. The
// wrapper guarantees N <= M, so a diagonal never holds more than N cells.
//
// Global tensors (C, Q, R, P, dC, dQ) use a diagonal-major layout: the
// cells of diagonal d are stored contiguously, ordered by row. Every global
// access of a diagonal is then coalesced, and the next diagonal's operands
// are prefetched into registers while the current one is computed.
//
// Forward (per cell (i, j), predecessors k in {(i-1,j-1), (i-1,j), (i,j-1)}):
//   R_ij = C_ij + SoftMin_gamma(R_k)
//   P_ij = Q_ij + sum_k q_k P_k,      q = softmax(-R_k / gamma)
//
// Backward, for L = gR * R_NM + gP * P_NM, successors s of cell k:
//   q_{k->s}  = exp((R_s - C_s - R_k) / gamma)
//   Pbar_k    = sum_s q_{k->s} Pbar_s
//   Rbar_k    = sum_s q_{k->s} [Rbar_s - Pbar_s (P_k - (P_s - Q_s)) / gamma]
//   dL/dC = Rbar, dL/dQ = Pbar.
//
// Only portable constructs are used (shared memory + __syncthreads), so the
// file builds unchanged under ROCm through PyTorch's automatic hipify.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <vector>

namespace {

constexpr int kMaxThreads = 256;
constexpr size_t kMaxStaticSmem = 48 * 1024;

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

__device__ __forceinline__ int diag_lo(int d, int M) { return max(0, d - (M - 1)); }
__device__ __forceinline__ int diag_hi(int d, int N) { return min(N - 1, d); }

template <typename scalar_t, int ROWS>
__global__ void udtw_forward_kernel(
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ Q,
    scalar_t* __restrict__ R,
    scalar_t* __restrict__ P,
    scalar_t* __restrict__ out_r,
    scalar_t* __restrict__ out_p,
    scalar_t* __restrict__ scratch,
    const int N,
    const int M,
    const scalar_t gamma,
    const scalar_t bandwidth) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int b = blockIdx.x;
  const long long off_b = (long long)b * N * M;
  C += off_b; Q += off_b; R += off_b; P += off_b;

  scalar_t* buf = scratch != nullptr
      ? scratch + (long long)b * 6 * N
      : reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* Rb = buf;          // [3][N]
  scalar_t* Pb = buf + 3 * N;  // [3][N]

  const scalar_t inf = (scalar_t)INFINITY;
  const scalar_t inv_g = (scalar_t)1 / gamma;
  const int D = N + M - 1;

  scalar_t c_nx[ROWS], q_nx[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int i = threadIdx.x + r * blockDim.x;
    if (i == 0) { c_nx[r] = C[0]; q_nx[r] = Q[0]; }
  }

  int off = 0;  // start of diagonal d in the diagonal-major layout
  for (int d = 0; d < D; ++d) {
    const int lo = diag_lo(d, M), hi = diag_hi(d, N);
    const int len = hi - lo + 1;

    scalar_t c_cur[ROWS], q_cur[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) { c_cur[r] = c_nx[r]; q_cur[r] = q_nx[r]; }

    if (d + 1 < D) {  // prefetch diagonal d+1
      const int lo1 = diag_lo(d + 1, M), hi1 = diag_hi(d + 1, N);
      const int off1 = off + len;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int i = threadIdx.x + r * blockDim.x;
        if (i >= lo1 && i <= hi1) {
          c_nx[r] = C[off1 + i - lo1];
          q_nx[r] = Q[off1 + i - lo1];
        }
      }
    }

    scalar_t* Rc = Rb + (d % 3) * N;
    scalar_t* Pc = Pb + (d % 3) * N;
    const scalar_t* R1 = Rb + ((d + 2) % 3) * N;  // diagonal d-1
    const scalar_t* P1 = Pb + ((d + 2) % 3) * N;
    const scalar_t* R2 = Rb + ((d + 1) % 3) * N;  // diagonal d-2
    const scalar_t* P2 = Pb + ((d + 1) % 3) * N;

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int i = threadIdx.x + r * blockDim.x;
      if (i < lo || i > hi) continue;
      const int j = d - i;
      const bool allowed =
          bandwidth <= (scalar_t)0 || fabs((scalar_t)(i - j)) <= bandwidth;
      scalar_t rv, pv;
      if (!allowed) {
        rv = inf; pv = 0;
      } else if (d == 0) {
        rv = c_cur[r]; pv = q_cur[r];
      } else {
        const scalar_t rd = (i > 0 && j > 0) ? R2[i - 1] : inf;
        const scalar_t ru = (i > 0) ? R1[i - 1] : inf;
        const scalar_t rl = (j > 0) ? R1[i] : inf;
        const scalar_t m = dmin(rd, dmin(ru, rl));
        if (m == inf) {
          rv = inf; pv = 0;
        } else {
          const scalar_t ed = dexp((m - rd) * inv_g);
          const scalar_t eu = dexp((m - ru) * inv_g);
          const scalar_t el = dexp((m - rl) * inv_g);
          const scalar_t s = ed + eu + el;
          const scalar_t pd = (i > 0 && j > 0) ? P2[i - 1] : (scalar_t)0;
          const scalar_t pu = (i > 0) ? P1[i - 1] : (scalar_t)0;
          const scalar_t pl = (j > 0) ? P1[i] : (scalar_t)0;
          rv = c_cur[r] + m - gamma * dlog(s);
          pv = q_cur[r] + (ed * pd + eu * pu + el * pl) / s;
        }
      }
      Rc[i] = rv; Pc[i] = pv;
      R[off + i - lo] = rv; P[off + i - lo] = pv;
    }
    __syncthreads();
    off += len;
  }

  if (threadIdx.x == 0) {
    const int last = (D - 1) % 3;
    out_r[b] = Rb[last * N + N - 1];
    out_p[b] = Pb[last * N + N - 1];
  }
}

template <typename scalar_t, int ROWS>
__global__ void udtw_backward_kernel(
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ R,
    const scalar_t* __restrict__ P,
    const scalar_t* __restrict__ grad_r,
    const scalar_t* __restrict__ grad_p,
    scalar_t* __restrict__ dC,
    scalar_t* __restrict__ dQ,
    scalar_t* __restrict__ scratch,
    const int N,
    const int M,
    const scalar_t gamma) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int b = blockIdx.x;
  const long long off_b = (long long)b * N * M;
  C += off_b; Q += off_b; R += off_b; P += off_b; dC += off_b; dQ += off_b;

  scalar_t* buf = scratch != nullptr
      ? scratch + (long long)b * 12 * N
      : reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* Gb = buf;          // Rbar      [3][N]
  scalar_t* Hb = buf + 3 * N;  // Pbar      [3][N]
  scalar_t* Eb = buf + 6 * N;  // R - C     [3][N]
  scalar_t* Ab = buf + 9 * N;  // P - Q     [3][N]

  const scalar_t inf = (scalar_t)INFINITY;
  const scalar_t inv_g = (scalar_t)1 / gamma;
  const scalar_t gr = grad_r[b];
  const scalar_t gp = grad_p[b];
  const int D = N + M - 1;

  // The last diagonal holds the single cell (N-1, M-1).
  scalar_t r_nx[ROWS], p_nx[ROWS], c_nx[ROWS], q_nx[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int i = threadIdx.x + r * blockDim.x;
    if (i == N - 1) {
      const int o = N * M - 1;
      r_nx[r] = R[o]; p_nx[r] = P[o]; c_nx[r] = C[o]; q_nx[r] = Q[o];
    }
  }

  int off = N * M - 1;  // start of diagonal d
  for (int d = D - 1; d >= 0; --d) {
    const int lo = diag_lo(d, M), hi = diag_hi(d, N);

    scalar_t r_cur[ROWS], p_cur[ROWS], c_cur[ROWS], q_cur[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      r_cur[r] = r_nx[r]; p_cur[r] = p_nx[r]; c_cur[r] = c_nx[r]; q_cur[r] = q_nx[r];
    }

    int off1 = 0;
    if (d > 0) {  // prefetch diagonal d-1
      const int lo1 = diag_lo(d - 1, M), hi1 = diag_hi(d - 1, N);
      off1 = off - (hi1 - lo1 + 1);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int i = threadIdx.x + r * blockDim.x;
        if (i >= lo1 && i <= hi1) {
          const int o = off1 + i - lo1;
          r_nx[r] = R[o]; p_nx[r] = P[o]; c_nx[r] = C[o]; q_nx[r] = Q[o];
        }
      }
    }

    const int c0 = (d % 3) * N;
    const int c1 = ((d + 1) % 3) * N;  // diagonal d+1
    const int c2 = ((d + 2) % 3) * N;  // diagonal d+2

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int i = threadIdx.x + r * blockDim.x;
      if (i < lo || i > hi) continue;
      const int j = d - i;
      const scalar_t rk = r_cur[r];
      const scalar_t pk = p_cur[r];
      scalar_t g = 0, h = 0;
      if (d == D - 1) {
        g = gr; h = gp;
      } else if (rk != inf) {
        if (i + 1 < N) {  // successor (i+1, j) on diagonal d+1
          const int s = c1 + i + 1;
          const scalar_t w = dexp((Eb[s] - rk) * inv_g);
          g += w * (Gb[s] - Hb[s] * (pk - Ab[s]) * inv_g);
          h += w * Hb[s];
        }
        if (j + 1 < M) {  // successor (i, j+1) on diagonal d+1
          const int s = c1 + i;
          const scalar_t w = dexp((Eb[s] - rk) * inv_g);
          g += w * (Gb[s] - Hb[s] * (pk - Ab[s]) * inv_g);
          h += w * Hb[s];
        }
        if (i + 1 < N && j + 1 < M) {  // successor (i+1, j+1) on diagonal d+2
          const int s = c2 + i + 1;
          const scalar_t w = dexp((Eb[s] - rk) * inv_g);
          g += w * (Gb[s] - Hb[s] * (pk - Ab[s]) * inv_g);
          h += w * Hb[s];
        }
      }
      Gb[c0 + i] = g;
      Hb[c0 + i] = h;
      Eb[c0 + i] = (rk == inf) ? -inf : rk - c_cur[r];
      Ab[c0 + i] = pk - q_cur[r];
      dC[off + i - lo] = g;
      dQ[off + i - lo] = h;
    }
    __syncthreads();
    off = off1;
  }
}

// ---------------------------------------------------------------------------
// Fused path: the cost/penalty matrices are built inside the kernel from
//   G = X Y^T [B, N*M] (diagonal-major), x2 = ||x_i||^2, y2 = ||y_j||^2,
//   vx = sigma_x^2, vy = sigma_y^2
// as  sq = max(x2_i + y2_j - 2 G_ij, 0),  var = max(0.5 (vx_i + vy_j), eps),
//     C = sq / var,  Q = beta * dlog(var)       (Eqs. 8 and 16).
// G and dG use the same diagonal-major layout as R and P, so every global
// access is coalesced; the next diagonal's operands are prefetched into
// registers. Shared memory stays O(N + M), which preserves full occupancy.
// (Staging a row-major G tile in shared memory, or walking it diagonally in
// global memory, were both measured slower on a 64 KiB-smem / 6 MiB-L2 GPU.)
// The backward pass returns dG, dx2, dy2, dvx, dvy; row sums stay in
// registers (each thread owns its rows) and column sums go to shared memory
// without atomics (a diagonal touches each column at most once).
// Clamp gradients follow torch.clamp_min (zero at the boundary).
// ---------------------------------------------------------------------------

template <typename scalar_t, int ROWS>
__global__ void udtw_fused_forward_kernel(
    const scalar_t* __restrict__ G,
    const scalar_t* __restrict__ x2,
    const scalar_t* __restrict__ y2,
    const scalar_t* __restrict__ vx,
    const scalar_t* __restrict__ vy,
    scalar_t* __restrict__ R,
    scalar_t* __restrict__ P,
    scalar_t* __restrict__ out_r,
    scalar_t* __restrict__ out_p,
    const int N,
    const int M,
    const scalar_t gamma,
    const scalar_t beta,
    const scalar_t eps,
    const scalar_t bandwidth) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int b = blockIdx.x;
  G += (long long)b * N * M;
  R += (long long)b * N * M;
  P += (long long)b * N * M;
  x2 += (long long)b * N; vx += (long long)b * N;
  y2 += (long long)b * M; vy += (long long)b * M;

  scalar_t* ys = reinterpret_cast<scalar_t*>(smem_raw);  // y2 [M]
  scalar_t* vys = ys + M;                                // vy [M]
  scalar_t* Rb = vys + M;                                // [3][N]
  scalar_t* Pb = Rb + 3 * N;                             // [3][N]

  for (int k = threadIdx.x; k < M; k += blockDim.x) {
    ys[k] = y2[k];
    vys[k] = vy[k];
  }
  scalar_t xr[ROWS], vr[ROWS], g_nx[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int i = threadIdx.x + r * blockDim.x;
    if (i < N) { xr[r] = x2[i]; vr[r] = vx[i]; }
    if (i == 0) g_nx[r] = G[0];
  }
  __syncthreads();

  const scalar_t inf = (scalar_t)INFINITY;
  const scalar_t inv_g = (scalar_t)1 / gamma;
  const int D = N + M - 1;

  int off = 0;
  for (int d = 0; d < D; ++d) {
    const int lo = diag_lo(d, M), hi = diag_hi(d, N);

    scalar_t g_cur[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) g_cur[r] = g_nx[r];
    if (d + 1 < D) {  // prefetch G on diagonal d+1
      const int lo1 = diag_lo(d + 1, M), hi1 = diag_hi(d + 1, N);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int i = threadIdx.x + r * blockDim.x;
        if (i >= lo1 && i <= hi1) g_nx[r] = G[off + (hi - lo + 1) + i - lo1];
      }
    }

    scalar_t* Rc = Rb + (d % 3) * N;
    scalar_t* Pc = Pb + (d % 3) * N;
    const scalar_t* R1 = Rb + ((d + 2) % 3) * N;
    const scalar_t* P1 = Pb + ((d + 2) % 3) * N;
    const scalar_t* R2 = Rb + ((d + 1) % 3) * N;
    const scalar_t* P2 = Pb + ((d + 1) % 3) * N;

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int i = threadIdx.x + r * blockDim.x;
      if (i < lo || i > hi) continue;
      const int j = d - i;
      scalar_t rv, pv;
      const bool allowed =
          bandwidth <= (scalar_t)0 || fabs((scalar_t)(i - j)) <= bandwidth;
      if (!allowed) {
        rv = inf; pv = 0;
      } else {
        scalar_t sq = xr[r] + ys[j] - (scalar_t)2 * g_cur[r];
        sq = sq > (scalar_t)0 ? sq : (scalar_t)0;
        scalar_t var = (scalar_t)0.5 * (vr[r] + vys[j]);
        var = var > eps ? var : eps;
        const scalar_t c = sq / var;
        const scalar_t q = beta * dlog(var);
        if (d == 0) {
          rv = c; pv = q;
        } else {
          const scalar_t rd = (i > 0 && j > 0) ? R2[i - 1] : inf;
          const scalar_t ru = (i > 0) ? R1[i - 1] : inf;
          const scalar_t rl = (j > 0) ? R1[i] : inf;
          const scalar_t m = dmin(rd, dmin(ru, rl));
          if (m == inf) {
            rv = inf; pv = 0;
          } else {
            const scalar_t ed = dexp((m - rd) * inv_g);
            const scalar_t eu = dexp((m - ru) * inv_g);
            const scalar_t el = dexp((m - rl) * inv_g);
            const scalar_t s = ed + eu + el;
            const scalar_t pd = (i > 0 && j > 0) ? P2[i - 1] : (scalar_t)0;
            const scalar_t pu = (i > 0) ? P1[i - 1] : (scalar_t)0;
            const scalar_t pl = (j > 0) ? P1[i] : (scalar_t)0;
            rv = c + m - gamma * dlog(s);
            pv = q + (ed * pd + eu * pu + el * pl) / s;
          }
        }
      }
      Rc[i] = rv; Pc[i] = pv;
      R[off + i - lo] = rv; P[off + i - lo] = pv;
    }
    __syncthreads();
    off += hi - lo + 1;
  }

  if (threadIdx.x == 0) {
    const int last = (D - 1) % 3;
    out_r[b] = Rb[last * N + N - 1];
    out_p[b] = Pb[last * N + N - 1];
  }
}

template <typename scalar_t, int ROWS>
__global__ void udtw_fused_backward_kernel(
    const scalar_t* __restrict__ G,
    const scalar_t* __restrict__ x2,
    const scalar_t* __restrict__ y2,
    const scalar_t* __restrict__ vx,
    const scalar_t* __restrict__ vy,
    const scalar_t* __restrict__ R,
    const scalar_t* __restrict__ P,
    const scalar_t* __restrict__ grad_r,
    const scalar_t* __restrict__ grad_p,
    scalar_t* __restrict__ dG,
    scalar_t* __restrict__ dx2,
    scalar_t* __restrict__ dy2,
    scalar_t* __restrict__ dvx,
    scalar_t* __restrict__ dvy,
    const int N,
    const int M,
    const scalar_t gamma,
    const scalar_t beta,
    const scalar_t eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const int b = blockIdx.x;
  G += (long long)b * N * M;
  dG += (long long)b * N * M;
  R += (long long)b * N * M;
  P += (long long)b * N * M;
  x2 += (long long)b * N; vx += (long long)b * N;
  dx2 += (long long)b * N; dvx += (long long)b * N;
  y2 += (long long)b * M; vy += (long long)b * M;
  dy2 += (long long)b * M; dvy += (long long)b * M;

  scalar_t* ys = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* vys = ys + M;
  scalar_t* dys = vys + M;
  scalar_t* dvys = dys + M;
  scalar_t* Gb = dvys + M;     // Rbar   [3][N]
  scalar_t* Hb = Gb + 3 * N;   // Pbar   [3][N]
  scalar_t* Eb = Hb + 3 * N;   // R - C  [3][N]
  scalar_t* Ab = Eb + 3 * N;   // P - Q  [3][N]

  for (int k = threadIdx.x; k < M; k += blockDim.x) {
    ys[k] = y2[k]; vys[k] = vy[k];
    dys[k] = 0; dvys[k] = 0;
  }
  scalar_t xr[ROWS], vr[ROWS], dxr[ROWS], dvr[ROWS];
  scalar_t r_nx[ROWS], p_nx[ROWS], g_nx[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int i = threadIdx.x + r * blockDim.x;
    dxr[r] = 0; dvr[r] = 0;
    if (i < N) { xr[r] = x2[i]; vr[r] = vx[i]; }
    if (i == N - 1) {
      r_nx[r] = R[N * M - 1]; p_nx[r] = P[N * M - 1]; g_nx[r] = G[N * M - 1];
    }
  }
  __syncthreads();

  const scalar_t inf = (scalar_t)INFINITY;
  const scalar_t inv_g = (scalar_t)1 / gamma;
  const scalar_t gr = grad_r[b];
  const scalar_t gp = grad_p[b];
  const int D = N + M - 1;

  int off = N * M - 1;
  for (int d = D - 1; d >= 0; --d) {
    const int lo = diag_lo(d, M), hi = diag_hi(d, N);

    scalar_t r_cur[ROWS], p_cur[ROWS], g_cur[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) { r_cur[r] = r_nx[r]; p_cur[r] = p_nx[r]; g_cur[r] = g_nx[r]; }

    int off1 = 0;
    if (d > 0) {  // prefetch diagonal d-1
      const int lo1 = diag_lo(d - 1, M), hi1 = diag_hi(d - 1, N);
      off1 = off - (hi1 - lo1 + 1);
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int i = threadIdx.x + r * blockDim.x;
        if (i >= lo1 && i <= hi1) {
          r_nx[r] = R[off1 + i - lo1];
          p_nx[r] = P[off1 + i - lo1];
          g_nx[r] = G[off1 + i - lo1];
        }
      }
    }

    const int c0 = (d % 3) * N;
    const int c1 = ((d + 1) % 3) * N;
    const int c2 = ((d + 2) % 3) * N;

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int i = threadIdx.x + r * blockDim.x;
      if (i < lo || i > hi) continue;
      const int j = d - i;
      const scalar_t rk = r_cur[r];
      const scalar_t pk = p_cur[r];

      const scalar_t sq_raw = xr[r] + ys[j] - (scalar_t)2 * g_cur[r];
      const scalar_t sq = sq_raw > (scalar_t)0 ? sq_raw : (scalar_t)0;
      const scalar_t var_raw = (scalar_t)0.5 * (vr[r] + vys[j]);
      const scalar_t var = var_raw > eps ? var_raw : eps;
      const scalar_t c = sq / var;
      const scalar_t q = beta * dlog(var);

      scalar_t g = 0, h = 0;
      if (d == D - 1) {
        g = gr; h = gp;
      } else if (rk != inf) {
        if (i + 1 < N) {
          const int s = c1 + i + 1;
          const scalar_t w = dexp((Eb[s] - rk) * inv_g);
          g += w * (Gb[s] - Hb[s] * (pk - Ab[s]) * inv_g);
          h += w * Hb[s];
        }
        if (j + 1 < M) {
          const int s = c1 + i;
          const scalar_t w = dexp((Eb[s] - rk) * inv_g);
          g += w * (Gb[s] - Hb[s] * (pk - Ab[s]) * inv_g);
          h += w * Hb[s];
        }
        if (i + 1 < N && j + 1 < M) {
          const int s = c2 + i + 1;
          const scalar_t w = dexp((Eb[s] - rk) * inv_g);
          g += w * (Gb[s] - Hb[s] * (pk - Ab[s]) * inv_g);
          h += w * Hb[s];
        }
      }
      Gb[c0 + i] = g;
      Hb[c0 + i] = h;
      Eb[c0 + i] = (rk == inf) ? -inf : rk - c;
      Ab[c0 + i] = pk - q;

      // chain rule through C = sq / var and Q = beta * dlog(var)
      const scalar_t dsq = sq_raw > (scalar_t)0 ? g / var : (scalar_t)0;
      const scalar_t dvar = var_raw > eps ? (beta * h - g * c) / var : (scalar_t)0;
      dG[off + i - lo] = (scalar_t)-2 * dsq;
      dxr[r] += dsq;
      dvr[r] += (scalar_t)0.5 * dvar;
      dys[j] += dsq;
      dvys[j] += (scalar_t)0.5 * dvar;
    }
    __syncthreads();
    off = off1;
  }

  for (int k = threadIdx.x; k < M; k += blockDim.x) {
    dy2[k] = dys[k]; dvy[k] = dvys[k];
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    const int i = threadIdx.x + r * blockDim.x;
    if (i < N) { dx2[i] = dxr[r]; dvx[i] = dvr[r]; }
  }
}

size_t fused_smem_bytes(int N, int M, bool backward, size_t elem) {
  const size_t n = (backward ? 4 : 2) * (size_t)M + (backward ? 12 : 6) * (size_t)N;
  return n * elem;
}

struct LaunchShape {
  int threads;
  int rows;
};

LaunchShape pick_shape(int N) {
  int threads = ((N + 31) / 32) * 32;
  threads = threads < 32 ? 32 : (threads > kMaxThreads ? kMaxThreads : threads);
  int rows = (N + threads - 1) / threads;
  int p = 1;
  while (p < rows) p *= 2;
  TORCH_CHECK(p <= 64, "sequence too long for the CUDA kernel (N <= ",
              64 * kMaxThreads, " after transposition)");
  return {threads, p};
}

#define UDTW_DISPATCH_ROWS(ROWS_VAR, ...)                          \
  switch (ROWS_VAR) {                                              \
    case 1: { constexpr int ROWS = 1; __VA_ARGS__(); break; }      \
    case 2: { constexpr int ROWS = 2; __VA_ARGS__(); break; }      \
    case 4: { constexpr int ROWS = 4; __VA_ARGS__(); break; }      \
    case 8: { constexpr int ROWS = 8; __VA_ARGS__(); break; }      \
    case 16: { constexpr int ROWS = 16; __VA_ARGS__(); break; }    \
    case 32: { constexpr int ROWS = 32; __VA_ARGS__(); break; }    \
    default: { constexpr int ROWS = 64; __VA_ARGS__(); break; }    \
  }

void check_input(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 2, name, " must have shape [B, N*M] (diagonal-major)");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> udtw_forward(
    torch::Tensor C, torch::Tensor Q, int64_t N, int64_t M,
    double gamma, double bandwidth) {
  check_input(C, "cost");
  check_input(Q, "penalty");
  TORCH_CHECK(C.sizes() == Q.sizes(), "cost and penalty must have identical shapes");
  TORCH_CHECK(C.scalar_type() == Q.scalar_type(), "cost and penalty must share a dtype");
  TORCH_CHECK(C.size(1) == N * M, "cost does not match N*M");
  TORCH_CHECK(N <= M, "internal: expected N <= M");
  const c10::cuda::OptionalCUDAGuard guard(C.device());

  const int B = C.size(0);
  auto R = torch::empty_like(C);
  auto P = torch::empty_like(C);
  auto out_r = torch::empty({B}, C.options());
  auto out_p = torch::empty({B}, C.options());
  if (B == 0) return {out_r, out_p, R, P};

  const size_t smem = 6 * (size_t)N * C.element_size();
  torch::Tensor scratch;
  if (smem > kMaxStaticSmem) scratch = torch::empty({(long long)B * 6 * N}, C.options());
  const LaunchShape shape = pick_shape(N);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(C.scalar_type(), "udtw_forward", [&] {
    UDTW_DISPATCH_ROWS(shape.rows, [&] {
      udtw_forward_kernel<scalar_t, ROWS>
          <<<B, shape.threads, scratch.defined() ? 0 : smem, stream>>>(
              C.data_ptr<scalar_t>(), Q.data_ptr<scalar_t>(),
              R.data_ptr<scalar_t>(), P.data_ptr<scalar_t>(),
              out_r.data_ptr<scalar_t>(), out_p.data_ptr<scalar_t>(),
              scratch.defined() ? scratch.data_ptr<scalar_t>() : nullptr,
              (int)N, (int)M, (scalar_t)gamma, (scalar_t)bandwidth);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_p, R, P};
}

std::vector<torch::Tensor> udtw_backward(
    torch::Tensor C, torch::Tensor Q, torch::Tensor R, torch::Tensor P,
    torch::Tensor grad_r, torch::Tensor grad_p, int64_t N, int64_t M, double gamma) {
  check_input(C, "cost");
  check_input(Q, "penalty");
  check_input(R, "R");
  check_input(P, "P");
  TORCH_CHECK(grad_r.is_contiguous() && grad_p.is_contiguous(), "grads must be contiguous");
  const c10::cuda::OptionalCUDAGuard guard(C.device());

  const int B = C.size(0);
  auto dC = torch::empty_like(C);
  auto dQ = torch::empty_like(C);
  if (B == 0) return {dC, dQ};

  const size_t smem = 12 * (size_t)N * C.element_size();
  torch::Tensor scratch;
  if (smem > kMaxStaticSmem) scratch = torch::empty({(long long)B * 12 * N}, C.options());
  const LaunchShape shape = pick_shape(N);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(C.scalar_type(), "udtw_backward", [&] {
    UDTW_DISPATCH_ROWS(shape.rows, [&] {
      udtw_backward_kernel<scalar_t, ROWS>
          <<<B, shape.threads, scratch.defined() ? 0 : smem, stream>>>(
              C.data_ptr<scalar_t>(), Q.data_ptr<scalar_t>(),
              R.data_ptr<scalar_t>(), P.data_ptr<scalar_t>(),
              grad_r.data_ptr<scalar_t>(), grad_p.data_ptr<scalar_t>(),
              dC.data_ptr<scalar_t>(), dQ.data_ptr<scalar_t>(),
              scratch.defined() ? scratch.data_ptr<scalar_t>() : nullptr,
              (int)N, (int)M, (scalar_t)gamma);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dC, dQ};
}

int64_t fused_smem_limit() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  int optin = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  return optin;
}

bool fused_fits(int64_t N, int64_t M, int64_t elem) {
  return (int64_t)fused_smem_bytes(N, M, true, elem) <= fused_smem_limit();
}

template <typename Kernel>
void allow_smem(Kernel kernel, size_t smem) {
  if (smem > kMaxStaticSmem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  }
}

void check_vec(const torch::Tensor& t, int64_t B, int64_t L, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
  TORCH_CHECK(t.dim() == 2 && t.size(0) == B && t.size(1) == L, name, " has the wrong shape");
}

std::vector<torch::Tensor> udtw_fused_forward(
    torch::Tensor G, torch::Tensor x2, torch::Tensor y2, torch::Tensor vx, torch::Tensor vy,
    double gamma, double beta, double eps, double bandwidth) {
  TORCH_CHECK(G.is_cuda() && G.dim() == 2 && G.is_contiguous(),
              "G must be [B, N*M] (diagonal-major) contiguous CUDA");
  const int B = G.size(0), N = x2.size(1), M = y2.size(1);
  TORCH_CHECK(G.size(1) == (int64_t)N * M, "G does not match N*M");
  TORCH_CHECK(N <= M, "internal: expected N <= M");
  check_vec(x2, B, N, "x2"); check_vec(vx, B, N, "vx");
  check_vec(y2, B, M, "y2"); check_vec(vy, B, M, "vy");
  const c10::cuda::OptionalCUDAGuard guard(G.device());

  auto R = torch::empty({B, (long long)N * M}, G.options());
  auto P = torch::empty_like(R);
  auto out_r = torch::empty({B}, G.options());
  auto out_p = torch::empty({B}, G.options());
  if (B == 0) return {out_r, out_p, R, P};

  const size_t smem = fused_smem_bytes(N, M, false, G.element_size());
  TORCH_CHECK((int64_t)smem <= fused_smem_limit(), "problem too large for the fused kernel");
  const LaunchShape shape = pick_shape(N);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(G.scalar_type(), "udtw_fused_forward", [&] {
    UDTW_DISPATCH_ROWS(shape.rows, [&] {
      auto kernel = udtw_fused_forward_kernel<scalar_t, ROWS>;
      allow_smem(kernel, smem);
      kernel<<<B, shape.threads, smem, stream>>>(
          G.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), y2.data_ptr<scalar_t>(),
          vx.data_ptr<scalar_t>(), vy.data_ptr<scalar_t>(),
          R.data_ptr<scalar_t>(), P.data_ptr<scalar_t>(),
          out_r.data_ptr<scalar_t>(), out_p.data_ptr<scalar_t>(),
          N, M, (scalar_t)gamma, (scalar_t)beta, (scalar_t)eps, (scalar_t)bandwidth);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_r, out_p, R, P};
}

std::vector<torch::Tensor> udtw_fused_backward(
    torch::Tensor G, torch::Tensor x2, torch::Tensor y2, torch::Tensor vx, torch::Tensor vy,
    torch::Tensor R, torch::Tensor P, torch::Tensor grad_r, torch::Tensor grad_p,
    double gamma, double beta, double eps) {
  const int B = G.size(0), N = x2.size(1), M = y2.size(1);
  TORCH_CHECK(R.is_contiguous() && P.is_contiguous(), "R and P must be contiguous");
  TORCH_CHECK(grad_r.is_contiguous() && grad_p.is_contiguous(), "grads must be contiguous");
  const c10::cuda::OptionalCUDAGuard guard(G.device());

  auto dG = torch::empty_like(G);
  auto dx2 = torch::empty_like(x2);
  auto dy2 = torch::empty_like(y2);
  auto dvx = torch::empty_like(vx);
  auto dvy = torch::empty_like(vy);
  if (B == 0) return {dG, dx2, dy2, dvx, dvy};

  const size_t smem = fused_smem_bytes(N, M, true, G.element_size());
  TORCH_CHECK((int64_t)smem <= fused_smem_limit(), "problem too large for the fused kernel");
  const LaunchShape shape = pick_shape(N);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(G.scalar_type(), "udtw_fused_backward", [&] {
    UDTW_DISPATCH_ROWS(shape.rows, [&] {
      auto kernel = udtw_fused_backward_kernel<scalar_t, ROWS>;
      allow_smem(kernel, smem);
      kernel<<<B, shape.threads, smem, stream>>>(
          G.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), y2.data_ptr<scalar_t>(),
          vx.data_ptr<scalar_t>(), vy.data_ptr<scalar_t>(),
          R.data_ptr<scalar_t>(), P.data_ptr<scalar_t>(),
          grad_r.data_ptr<scalar_t>(), grad_p.data_ptr<scalar_t>(),
          dG.data_ptr<scalar_t>(), dx2.data_ptr<scalar_t>(), dy2.data_ptr<scalar_t>(),
          dvx.data_ptr<scalar_t>(), dvy.data_ptr<scalar_t>(),
          N, M, (scalar_t)gamma, (scalar_t)beta, (scalar_t)eps);
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dG, dx2, dy2, dvx, dvy};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &udtw_forward, "uDTW forward (CUDA/HIP, diagonal-major layout)");
  m.def("backward", &udtw_backward, "uDTW backward (CUDA/HIP, diagonal-major layout)");
  m.def("fused_forward", &udtw_fused_forward, "uDTW forward with in-kernel cost construction");
  m.def("fused_backward", &udtw_fused_backward, "uDTW backward with in-kernel cost construction");
  m.def("fused_fits", &fused_fits, "whether (N, M, elem_size) fits the fused kernel");
}
