#include <cstdint>
#include <mma.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include <cutf/thread.hpp>
#include <gemm_core/gemm_core.hpp>

#include <tsqr_tc/batchedqr.hpp>
#include "utils.hpp"
#include <tsqr_tc/detail/macro.hpp>

//#define MTK_DEBUG
//#define MTK_CLOCK_BREAKDOWN

#ifdef MTK_DEBUG
#include <cutf/debug/matrix.hpp>
#define MTK_DEBUG_PRINT_MATRIX(ptr, m, n, ldm, name) \
	__syncthreads(); \
	if (threadIdx.x + blockIdx.x == 0) cutf::debug::print::print_numpy_matrix(ptr, m, n, ldm, name); \
	__syncthreads();
#define MTK_DEBUG_CALL_FUNC(func) \
	__syncthreads(); \
	if (threadIdx.x + blockIdx.x == 0) {func;} \
	__syncthreads();
#else
#define MTK_DEBUG_PRINT_MATRIX(ptr, m, n, ldm, name)
#define MTK_DEBUG_CALL_FUNC(func)
#endif

#ifdef MTK_CLOCK_BREAKDOWN
#include <cutf/debug/clock_breakdown.hpp>
#define MTK_CLOCK_BREAKDOWN_INIT(n) CUTF_CLOCK_BREAKDOWN_INIT(n)
#define MTK_CLOCK_BREAKDOWN_RECORD(n) CUTF_CLOCK_BREAKDOWN_RECORD(n)
#define MTK_CLOCK_BREAKDOWN_DURATION(m, n) CUTF_CLOCK_BREAKDOWN_DURATION(m, n)
#else
#define MTK_CLOCK_BREAKDOWN_INIT(n)
#define MTK_CLOCK_BREAKDOWN_RECORD(n)
#define MTK_CLOCK_BREAKDOWN_DURATION(m, n)
#endif

namespace {

constexpr unsigned warp_size = 32u;

// This function computes L2-norm ^2 of a given vector(array).
// Restrictions:
// - size % warp_size == 0
template <class COMPUTE_T, class T>
__device__ COMPUTE_T compute_norm2(const T* const ptr, const unsigned size) {
	auto norm2 = cutf::type::cast<COMPUTE_T>(0.0f);
	for (unsigned i = 0; i < size; i += warp_size) {
		const auto v = cutf::type::cast<COMPUTE_T>(ptr[i + cutf::thread::get_lane_id()]);
		norm2 += v * v;
	}
	for(auto mask = (warp_size >> 1); mask > 0; mask >>= 1) {
		norm2 += __shfl_xor_sync(0xffffffff, norm2, mask);
	}
	return norm2;
}

// This function computes `tmp = y^t * A`.
// Restrictions:
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_0_hmma(
		float* const smem_reduction,
		const float* const smem_y,
		const float* const smem_A,
		const float t
		) {
	constexpr unsigned frag_dim = warp_size;
	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a   , frag_dim, mtk::tsqr_tc::utils::min_fragment_n<compute_mode>, frag_dim, nvcuda::wmma::row_major>::type frag_yt;
	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b   , frag_dim, mtk::tsqr_tc::utils::min_fragment_n<compute_mode>, frag_dim, nvcuda::wmma::col_major>::type frag_A;
	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, frag_dim, mtk::tsqr_tc::utils::min_fragment_n<compute_mode>, frag_dim>::type frag_ytA;

	mtk::wmma::tcec::fill_zero(frag_yt);
	mtk::wmma::tcec::load_vector(frag_yt, smem_y + (threadIdx.x & 0xffffffe0u));
	mtk::wmma::tcec::load_matrix_sync(frag_A, smem_A + (threadIdx.x & 0xffffffe0u), smem_ldm, false);

	mtk::wmma::tcec::mma_sync(frag_ytA, frag_yt, frag_A);

	mtk::wmma::tcec::store_vector(smem_reduction + smem_n * (threadIdx.x >> 5), frag_ytA, -t, nvcuda::wmma::mem_row_major);

	// Accumulate
	__syncthreads();
	MTK_DEBUG_PRINT_MATRIX(smem_reduction, 1, smem_m, 1, "tmp (before accumulating)");
	mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_reduction, smem_n);
	MTK_DEBUG_PRINT_MATRIX(smem_reduction, 1, smem_m, 1, "tmp (accumulated)");
}

template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_0_notc(
		float* const smem_reduction,
		const float* const smem_y,
		const float* const smem_A,
		const float t
		) {
	constexpr unsigned frag_dim = warp_size;
	__syncthreads();
	mtk::gemm_core::gevm_core16x16<frag_dim, false>(
			smem_reduction + smem_n * (threadIdx.x >> 5),
			smem_y + (threadIdx.x & 0xffffffe0u),
			smem_A + (threadIdx.x & 0xffffffe0u),
			smem_ldm, threadIdx.x & 0x1f);
	if ((threadIdx.x & 0xf) < smem_n) {
		*(smem_reduction + smem_n * (threadIdx.x >> 5) + (threadIdx.x & 0xf)) *= -t;
	}
	// Accumulate
	__syncthreads();
	MTK_DEBUG_PRINT_MATRIX(smem_reduction, 1, smem_m, 1, "tmp (before accumulating)");
	mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_reduction, smem_n);
	MTK_DEBUG_PRINT_MATRIX(smem_reduction, 1, smem_m, 1, "tmp (accumulated)");
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_0(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_reduction_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_y_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_a_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type t 
		) {
	MTK_DEBUG_CALL_FUNC(printf("# --> %s\n", __func__));
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_no_tc) {
		compute_reflection_0_notc<smem_m, smem_n, smem_ldm>(smem_reduction_ptr, smem_y_ptr, smem_a_ptr, t);
	} else {
		compute_reflection_0_hmma<compute_mode, smem_m, smem_n, smem_ldm>(smem_reduction_ptr, smem_y_ptr, smem_a_ptr, t);
	}
	MTK_DEBUG_CALL_FUNC(printf("# <-- %s\n", __func__));
}

// This function computes `A = A -2t * y * tmp`.
// Restrictions:
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_1_hmma(
		float* const smem_A_ptr,
		float* const smem_reduction_ptr,
		const float* const smem_y_ptr
		) {
	constexpr unsigned num_col_block = warp_size / smem_n;

	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b, smem_n, smem_n, smem_n, nvcuda::wmma::row_major>::type frag_tmp;
	mtk::wmma::tcec::fill_zero(frag_tmp);
	mtk::wmma::tcec::load_vector(frag_tmp, smem_reduction_ptr);

	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a, smem_n, smem_n, smem_n, nvcuda::wmma::col_major>::type frag_y;
	mtk::wmma::tcec::fill_zero(frag_y);

	for (unsigned i = 0; i < num_col_block; i++) {
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, smem_n, smem_n, smem_n>::type frag_A;

		auto y_ptr = smem_y_ptr + (threadIdx.x & 0xffffffe0u) + i * smem_n;
		mtk::wmma::tcec::load_vector(frag_y, y_ptr);

		auto A_ptr = smem_A_ptr + (threadIdx.x & 0xffffffe0u);
		mtk::wmma::tcec::load_matrix_sync(frag_A, A_ptr + i * smem_n, smem_ldm, nvcuda::wmma::mem_col_major, false);

		mtk::wmma::tcec::mma_sync(frag_A, frag_y, frag_tmp, frag_A);
		mtk::wmma::tcec::store_matrix_sync(A_ptr + i * smem_n, frag_A, smem_ldm, nvcuda::wmma::mem_col_major, false);
	}
}

template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_1_notc(
		float* const smem_A_ptr,
		float* const smem_reduction_ptr,
		const float* const smem_y_ptr
		) {
	for (unsigned i = 0; i < smem_n; i++) {
		smem_A_ptr[i * smem_ldm + threadIdx.x] += smem_y_ptr[threadIdx.x] * smem_reduction_ptr[i];
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_1(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_A_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_reduction_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_y_ptr
		) {
	MTK_DEBUG_CALL_FUNC(printf("# --> %s\n", __func__));
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_no_tc) {
		compute_reflection_1_notc<smem_m, smem_n, smem_ldm>(smem_A_ptr, smem_reduction_ptr, smem_y_ptr);
	} else {
		compute_reflection_1_hmma<compute_mode, smem_m, smem_n, smem_ldm>(smem_A_ptr, smem_reduction_ptr, smem_y_ptr);
	}
	MTK_DEBUG_CALL_FUNC(printf("# <-- %s\n", __func__));
}

// This function computes `w = (I - W * Y^t)y`.
// Restrictions:
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_w_hmma(
		float* const smem_W_ptr,
		float* const smem_reduction_ptr,
		const float* const smem_Y_ptr,
		const float* const smem_t_ptr,
		const std::size_t m, const std::size_t real_block_n
		) {
	constexpr unsigned frag_dim = warp_size;
	// Compute YtY

	{
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a, smem_n, smem_n, frag_dim, nvcuda::wmma::row_major>::type frag_Yt;
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b, smem_n, smem_n, frag_dim, nvcuda::wmma::col_major>::type frag_Y;
		mtk::wmma::tcec::load_matrix_sync(frag_Yt, smem_Y_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);
		mtk::wmma::tcec::load_matrix_sync(frag_Y , smem_Y_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, smem_n, smem_n, frag_dim>::type frag_YtY;
		mtk::wmma::tcec::mma_sync(frag_YtY, frag_Yt, frag_Y);
		mtk::wmma::tcec::store_matrix_sync(smem_reduction_ptr + (threadIdx.x / warp_size) * smem_n * smem_n, frag_YtY, smem_n, nvcuda::wmma::mem_col_major, false);
	}

	// Accumulate
	__syncthreads();
	mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_reduction_ptr, smem_n * smem_n);
	MTK_DEBUG_PRINT_MATRIX(smem_reduction_ptr, smem_n, smem_n, smem_n, "YtY");

	// Compute W
	if (threadIdx.x < m) {
		for (std::size_t tn = 1; tn < real_block_n; tn++) {
			const auto t = smem_t_ptr[tn];
			float v = 0;
			for (std::size_t sn = 0; sn < tn; sn++) {
				v += smem_W_ptr[sn * smem_ldm + threadIdx.x] * t * smem_reduction_ptr[sn * smem_n + tn];
			}
			smem_W_ptr[tn * smem_ldm + threadIdx.x] += -v;
		}
	}
}

template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_w_notc(
		float* const smem_W_ptr,
		float* const smem_reduction_ptr,
		const float* const smem_Y_ptr,
		const float* const smem_t_ptr,
		const std::size_t m, const std::size_t real_block_n
		) {
	constexpr unsigned frag_dim = warp_size;
	// Compute YtY

	mtk::gemm_core::gemm_core16x16<frag_dim, 'T', 'N'>(
			1.0f,
			smem_Y_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm,
			smem_Y_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm,
			0.0f,
			smem_reduction_ptr + (threadIdx.x / warp_size) * smem_n * smem_n, smem_n
			);

	// Accumulate
	__syncthreads();
	mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_reduction_ptr, smem_n * smem_n);
	MTK_DEBUG_PRINT_MATRIX(smem_reduction_ptr, smem_n, smem_n, smem_n, "YtY");

	// Compute W
	if (threadIdx.x < m) {
		for (std::size_t tn = 1; tn < real_block_n; tn++) {
			const auto t = smem_t_ptr[tn];
			float v = 0;
			for (std::size_t sn = 0; sn < tn; sn++) {
				v += smem_W_ptr[sn * smem_ldm + threadIdx.x] * t * smem_reduction_ptr[sn * smem_n + tn];
			}
			smem_W_ptr[tn * smem_ldm + threadIdx.x] += -v;
		}
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_w(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_W_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_reduction_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_Y_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_t_ptr,
		const std::size_t m, const std::size_t real_block_n
		) {
	MTK_DEBUG_CALL_FUNC(printf("# --> %s\n", __func__));
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_no_tc) {
		compute_w_notc<smem_m, smem_n, smem_ldm>(
				smem_W_ptr,
				smem_reduction_ptr,
				smem_Y_ptr,
				smem_t_ptr,
				m, real_block_n
				);
	} else {
		compute_w_hmma<compute_mode, smem_m, smem_n, smem_ldm>(
				smem_W_ptr,
				smem_reduction_ptr,
				smem_Y_ptr,
				smem_t_ptr,
				m, real_block_n
				);
	}
	MTK_DEBUG_CALL_FUNC(printf("# <-- %s\n", __func__));
}

// This function computes `w = (I - W * Y^t)y`.
// Restrictions:
// - At first, smem_workspace_large_0_ptr must contain last Y block
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_base_w_hmma(
		float* const smem_workspace_large_0_ptr,
		float* const smem_workspace_large_1_ptr,
		float* const smem_workspace_large_2_ptr,
		float* const smem_workspace_small_ptr,
		const float* const smem_t_ptr,
		const float* const gmem_W_ptr, const std::size_t ldW,
		const float* const gmem_Y_ptr, const std::size_t ldY,
		const std::size_t m, const std::size_t n,
		const std::size_t real_block_n
		) {
	constexpr unsigned frag_dim = warp_size;

	if (n == 0) {
		for (std::size_t i = 0; i < real_block_n; i++) {
			smem_workspace_large_1_ptr[threadIdx.x + i * smem_ldm] = smem_t_ptr[i] * smem_workspace_large_0_ptr[threadIdx.x + i * smem_ldm];
		}
		return;
	}
	{
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b, smem_n, smem_n, frag_dim, nvcuda::wmma::col_major>::type frag_Yb;
		mtk::wmma::tcec::load_matrix_sync(frag_Yb, smem_workspace_large_0_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);

		// Compute YY <- Yg^t * Ys
		for (std::size_t bn = 0; bn < n; bn += smem_n) {
			typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a, smem_n, smem_n, frag_dim, nvcuda::wmma::row_major>::type frag_Yt;
			typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, smem_n, smem_n, frag_dim>::type frag_tmp;

			mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_Y_ptr + bn * ldY, ldY, m, smem_n);

			mtk::wmma::tcec::load_matrix_sync(frag_Yt, smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);
			mtk::wmma::tcec::mma_sync(frag_tmp, frag_Yt, frag_Yb);
			mtk::wmma::tcec::store_matrix_sync_with_mul(smem_workspace_small_ptr + (threadIdx.x / warp_size) * smem_n * smem_n, frag_tmp, smem_n, -1.0f, nvcuda::wmma::mem_col_major, false);

			// Accumulate
			__syncthreads();
			mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_workspace_large_2_ptr + bn * smem_n, smem_workspace_small_ptr, smem_n * smem_n);
			MTK_DEBUG_CALL_FUNC(printf("base YtY (%lu/%lu)\n", bn + 1, n));
			MTK_DEBUG_PRINT_MATRIX(smem_workspace_large_2_ptr + bn * smem_n, smem_n, smem_n, smem_n, "");
		}
	}

	// Compute Ws <- Ys - Wg * YY
	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, frag_dim, smem_n, smem_n>::type frag_w;
	mtk::wmma::tcec::load_matrix_sync(frag_w, smem_workspace_large_0_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, nvcuda::wmma::mem_col_major, false);
	__syncthreads();
	for (std::size_t bn = 0; bn < n; bn += smem_n) {
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b, frag_dim, smem_n, smem_n, nvcuda::wmma::col_major>::type frag_YY;
		mtk::wmma::tcec::load_matrix_sync(frag_YY, smem_workspace_large_2_ptr + bn * smem_n, smem_n, false);

		mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_W_ptr + bn * ldW, ldW, m, smem_n);

		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a, frag_dim, smem_n, smem_n, nvcuda::wmma::col_major>::type frag_W;
		mtk::wmma::tcec::load_matrix_sync(frag_W, smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);

		mtk::wmma::tcec::mma_sync(frag_w, frag_W, frag_YY, frag_w);
	}
	mtk::wmma::tcec::store_matrix_sync(smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), frag_w, smem_ldm, nvcuda::wmma::mem_col_major, false);
	if (threadIdx.x < m) {
		for (unsigned k = 0; k < real_block_n; k++) {
			smem_workspace_large_1_ptr[threadIdx.x + k * smem_ldm] *= smem_t_ptr[k];
		}
	}
	__syncthreads();
}

template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_base_w_notc(
		float* const smem_workspace_large_0_ptr,
		float* const smem_workspace_large_1_ptr,
		float* const smem_workspace_large_2_ptr,
		float* const smem_workspace_small_ptr,
		const float* const smem_t_ptr,
		const float* const gmem_W_ptr, const std::size_t ldW,
		const float* const gmem_Y_ptr, const std::size_t ldY,
		const std::size_t m, const std::size_t n,
		const std::size_t real_block_n
		) {
	constexpr unsigned frag_dim = warp_size;

	if (n == 0) {
		for (std::size_t i = 0; i < real_block_n; i++) {
			smem_workspace_large_1_ptr[threadIdx.x + i * smem_ldm] = smem_t_ptr[i] * smem_workspace_large_0_ptr[threadIdx.x + i * smem_ldm];
		}
		return;
	}
	{
		// Compute YY <- Yg^t * Ys
		for (std::size_t bn = 0; bn < n; bn += smem_n) {
			mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_Y_ptr + bn * ldY, ldY, m, smem_n);
			mtk::gemm_core::gemm_core16x16<frag_dim, 'T', 'N'>(
					-1.0f,
					smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm,
					smem_workspace_large_0_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm,
					0.0f,
					smem_workspace_small_ptr + (threadIdx.x / warp_size) * smem_n * smem_n, smem_n
					);

			// Accumulate
			__syncthreads();
			mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_workspace_large_2_ptr + bn * smem_n, smem_workspace_small_ptr, smem_n * smem_n);
			MTK_DEBUG_CALL_FUNC(printf("base YtY (%lu/%lu)\n", bn + 1, n));
			MTK_DEBUG_PRINT_MATRIX(smem_workspace_large_2_ptr + bn * smem_n, smem_n, smem_n, smem_n, "");
		}
	}

	// Compute Ws <- Ys - Wg * YY
	float reg[frag_dim * smem_n / warp_size];
	for (unsigned i = 0; i < smem_n; i++) {
		reg[i] = 0.0f;
	}
	for (std::size_t bn = 0; bn < n; bn += smem_n) {
		mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_W_ptr + bn * ldW, ldW, m, smem_n);
		for (unsigned i = 0; i < smem_n; i++) {
			for (unsigned k = 0; k < smem_n; k++) {
				const auto b_v = smem_workspace_large_2_ptr[bn * smem_n + i * smem_n + k];
				const auto a_v = smem_workspace_large_1_ptr[threadIdx.x + smem_ldm * k];
				reg[i] += a_v * b_v;
			}
		}
	}
	if (threadIdx.x < m) {
		for (unsigned k = 0; k < real_block_n; k++) {
			smem_workspace_large_1_ptr[threadIdx.x + k * smem_ldm] = smem_t_ptr[k] * (smem_workspace_large_0_ptr[threadIdx.x + k * smem_ldm] + reg[k]);
		}
	}
	__syncthreads();
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_base_w(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_large_0_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_large_1_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_large_2_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_small_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_t_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_W_ptr, const std::size_t ldW,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_Y_ptr, const std::size_t ldY,
		const std::size_t m, const std::size_t n,
		const std::size_t real_block_n
		) {
	MTK_DEBUG_CALL_FUNC(printf("# --> %s\n", __func__));
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_no_tc) {
		compute_base_w_notc<smem_m, smem_n, smem_ldm>(
				smem_workspace_large_0_ptr,
				smem_workspace_large_1_ptr,
				smem_workspace_large_2_ptr,
				smem_workspace_small_ptr,
				smem_t_ptr,
				gmem_W_ptr, ldW,
				gmem_Y_ptr, ldY,
				m, n, real_block_n
				);
	} else {
		compute_base_w_hmma<compute_mode, smem_m, smem_n, smem_ldm>(
				smem_workspace_large_0_ptr,
				smem_workspace_large_1_ptr,
				smem_workspace_large_2_ptr,
				smem_workspace_small_ptr,
				smem_t_ptr,
				gmem_W_ptr, ldW,
				gmem_Y_ptr, ldY,
				m, n, real_block_n
				);
	}
	MTK_DEBUG_CALL_FUNC(printf("# <-- %s\n", __func__));
}

// This function computes `A = (I - Y * W^t)A`.
// Restrictions:
// - At first, smem_workspace_large_0_ptr must contain A block
// - At first, smem_workspace_large_1_ptr must contain last W block
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void update_a_hmma(
		float* const smem_workspace_large_0_ptr,
		float* const smem_workspace_large_1_ptr,
		float* const smem_workspace_large_2_ptr,
		float* const smem_workspace_small_ptr,
		const float* const gmem_W_ptr, const std::size_t ldW,
		const float* const gmem_Y_ptr, const std::size_t ldY,
		const std::size_t m, const std::size_t n,
		const std::size_t real_block_n
		) {
	MTK_TSQR_TC_UNUSED(real_block_n);
	constexpr unsigned frag_dim = warp_size;

	if (n == 0) {
		return;
	}
	{
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b, smem_n, smem_n, frag_dim, nvcuda::wmma::col_major>::type frag_A;
		mtk::wmma::tcec::load_matrix_sync(frag_A, smem_workspace_large_0_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);

		// Compute WtA
		for (std::size_t bn = 0; bn < n; bn += smem_n) {
			typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a   , smem_n, smem_n, frag_dim, nvcuda::wmma::row_major>::type frag_Wt;
			typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, smem_n, smem_n, frag_dim>::type frag_tmp;

			mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_W_ptr + bn * ldW, ldW, m, smem_n);

			// Load Wt
			mtk::wmma::tcec::load_matrix_sync(frag_Wt, smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);

			mtk::wmma::tcec::mma_sync(frag_tmp, frag_Wt, frag_A);

			mtk::wmma::tcec::store_matrix_sync_with_mul(smem_workspace_small_ptr + (threadIdx.x / warp_size) * smem_n * smem_n, frag_tmp, smem_n, -1.0f, nvcuda::wmma::mem_col_major, false);

			// Accumulate
			__syncthreads();
			mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_workspace_large_2_ptr + bn * smem_n, smem_workspace_small_ptr, smem_n * smem_n);
			MTK_DEBUG_CALL_FUNC(printf("WtA (%lu/%lu)\n", bn + 1, n));
			MTK_DEBUG_PRINT_MATRIX(smem_workspace_large_2_ptr + bn * smem_n, smem_n, smem_n, smem_n, "");
		}
	}

	// Compute As <- As - Yg * WtA
	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, frag_dim, smem_n, smem_n>::type frag_A;
	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b   , frag_dim, smem_n, smem_n, nvcuda::wmma::col_major>::type frag_WtA;
	mtk::wmma::tcec::load_matrix_sync(frag_WtA, smem_workspace_large_2_ptr, smem_n, false);

	// Hand unrolling vvvvv
	mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_Y_ptr, ldY, m, smem_n);
	typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a   , frag_dim, smem_n, smem_n, nvcuda::wmma::col_major>::type frag_Y;
	// Load W
	mtk::wmma::tcec::load_matrix_sync(frag_Y, smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);
	mtk::wmma::tcec::load_matrix_sync(frag_A, smem_workspace_large_0_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, nvcuda::wmma::mem_col_major, false);
	mtk::wmma::tcec::mma_sync(frag_A, frag_Y, frag_WtA, frag_A);
	// Hand unrolling ^^^^^

	for (std::size_t bn = smem_n; bn < n; bn += smem_n) {
		mtk::wmma::tcec::load_matrix_sync(frag_WtA, smem_workspace_large_2_ptr + bn * smem_n, smem_n, false);

		mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_Y_ptr + bn * ldY, ldY, m, smem_n);
		// Load W
		mtk::wmma::tcec::load_matrix_sync(frag_Y, smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, false);
		mtk::wmma::tcec::mma_sync(frag_A, frag_Y, frag_WtA, frag_A);
	}
	mtk::wmma::tcec::store_matrix_sync(smem_workspace_large_0_ptr + (threadIdx.x & 0xffffffe0u), frag_A, smem_ldm, nvcuda::wmma::mem_col_major, false);
	__syncthreads();
}

template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void update_a_notc(
		float* const smem_workspace_large_0_ptr,
		float* const smem_workspace_large_1_ptr,
		float* const smem_workspace_large_2_ptr,
		float* const smem_workspace_small_ptr,
		const float* const gmem_W_ptr, const std::size_t ldW,
		const float* const gmem_Y_ptr, const std::size_t ldY,
		const std::size_t m, const std::size_t n,
		const std::size_t real_block_n
		) {
	constexpr unsigned frag_dim = warp_size;
	if (n == 0) {
		return;
	}
	{
		// Compute WtA
		for (std::size_t bn = 0; bn < n; bn += smem_n) {
			mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_W_ptr + bn * ldW, ldW, m, smem_n);
			mtk::gemm_core::gemm_core16x16<frag_dim, 'T', 'N'>(
					-1.0f,
					smem_workspace_large_1_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm,
					smem_workspace_large_0_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm,
					0.0f,
					smem_workspace_small_ptr + (threadIdx.x / warp_size) * smem_n * smem_n, smem_n
					);

			// Accumulate
			__syncthreads();
			mtk::tsqr_tc::utils::accumulate_vectors<smem_m>(smem_workspace_large_2_ptr + bn * smem_n, smem_workspace_small_ptr, smem_n * smem_n);
			MTK_DEBUG_CALL_FUNC(printf("WtA (%lu/%lu)\n", bn + 1, n));
			MTK_DEBUG_PRINT_MATRIX(smem_workspace_large_2_ptr + bn * smem_n, smem_n, smem_n, smem_n, "");
		}
	}

	// Compute As <- As - Yg * WtA
	float reg[frag_dim * smem_n / warp_size];
	for (unsigned i = 0; i < smem_n; i++) {
		reg[i] = 0.0f;
	}
	for (std::size_t bn = 0; bn < n; bn += smem_n) {
		mtk::tsqr_tc::utils::copy_matrix_g2s<smem_m, smem_n, smem_ldm>(smem_workspace_large_1_ptr, gmem_Y_ptr + bn * ldY, ldY, m, smem_n);
		for (unsigned i = 0; i < smem_n; i++) {
			for (unsigned k = 0; k < smem_n; k++) {
				const auto b_v = smem_workspace_large_2_ptr[bn * smem_n + i * smem_n + k];
				const auto a_v = smem_workspace_large_1_ptr[threadIdx.x + smem_ldm * k];
				reg[i] += a_v * b_v;
			}
		}
	}
	for (unsigned k = 0; k < real_block_n; k++) {
		smem_workspace_large_0_ptr[threadIdx.x + k * smem_ldm] += reg[k];
	}
	__syncthreads();
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void update_a(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_large_0_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_large_1_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_large_2_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_workspace_small_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_W_ptr, const std::size_t ldW,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_Y_ptr, const std::size_t ldY,
		const std::size_t m, const std::size_t n,
		const std::size_t real_block_n
		) {
	MTK_DEBUG_CALL_FUNC(printf("# --> %s\n", __func__));
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_no_tc) {
		update_a_notc<smem_m, smem_n, smem_ldm>(
				smem_workspace_large_0_ptr,
				smem_workspace_large_1_ptr,
				smem_workspace_large_2_ptr,
				smem_workspace_small_ptr,
				gmem_W_ptr, ldW,
				gmem_Y_ptr, ldY,
				m, n, real_block_n
				);
	} else {
		update_a_hmma<compute_mode, smem_m, smem_n, smem_ldm>(
				smem_workspace_large_0_ptr,
				smem_workspace_large_1_ptr,
				smem_workspace_large_2_ptr,
				smem_workspace_small_ptr,
				gmem_W_ptr, ldW,
				gmem_Y_ptr, ldY,
				m, n, real_block_n
				);
	}
	MTK_DEBUG_CALL_FUNC(printf("# <-- %s\n", __func__));
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
__device__ void qr_kernel(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_r_ptr, const std::size_t ldr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n
		) {
	using T = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	constexpr unsigned DIM_MAX_M = 256;
	constexpr unsigned DIM_BLOCK_N = 16;
	constexpr unsigned block_size = DIM_MAX_M;

	extern __shared__ T smem[];

	T* const smem_A_ptr = smem;
	T* const smem_W_ptr = smem_A_ptr + DIM_MAX_M * DIM_BLOCK_N;
	T* const smem_Y_ptr = smem_W_ptr + DIM_MAX_M * DIM_BLOCK_N;
	T* const smem_y_ptr = smem_Y_ptr + DIM_MAX_M * DIM_BLOCK_N;
	T* const smem_t_ptr = smem_y_ptr + DIM_MAX_M;
	T* const smem_tmp_ptr = smem_t_ptr + DIM_BLOCK_N;

	const unsigned num_n_blocks = (n + DIM_BLOCK_N - 1) / DIM_BLOCK_N;
	for (std::size_t n_block = 0; n_block < num_n_blocks; n_block++) {
		MTK_CLOCK_BREAKDOWN_INIT(6);
		MTK_CLOCK_BREAKDOWN_RECORD(0);

		const unsigned real_block_n = umin(DIM_BLOCK_N, n - DIM_BLOCK_N * n_block);
		mtk::tsqr_tc::utils::copy_matrix_g2s<block_size, DIM_BLOCK_N, DIM_MAX_M>(smem_A_ptr, gmem_a_ptr + lda * n_block * DIM_BLOCK_N, lda, m, real_block_n);
		MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, m, real_block_n, DIM_MAX_M, "A (Before updating)");
		update_a<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(
				smem_A_ptr,
				smem_W_ptr,
				smem_Y_ptr,
				smem_tmp_ptr,
				gmem_w_ptr, ldw,
				gmem_y_ptr, ldy,
				m, n_block * DIM_BLOCK_N,
				real_block_n
				);

		MTK_CLOCK_BREAKDOWN_RECORD(1);

		for (unsigned sn = 0; sn < real_block_n; sn++) {
			MTK_DEBUG_CALL_FUNC(printf("----------\n----- small n : %u\n----------\n", sn));
			MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, m, real_block_n, DIM_MAX_M, "Input A");
			const auto gn = n_block * DIM_BLOCK_N + sn;

			// Copy y from A
			auto yv = cutf::type::cast<T>(0.0f);
			if (threadIdx.x >= gn) {
				const auto index = DIM_MAX_M * sn + threadIdx.x;
				yv = smem_A_ptr[index];
			}
			smem_y_ptr[threadIdx.x] = yv;
			__syncthreads();
			MTK_DEBUG_PRINT_MATRIX(smem_y_ptr, 1, m, 1, "y (loaded)");

			// Compute norm2 of y and update y (y_i <- y_i +- norm(y);
			if ((threadIdx.x / warp_size) == gn / warp_size) {
				const auto norm2 = cutf::type::cast<T>(compute_norm2<float>(smem_y_ptr, DIM_MAX_M));
				if (cutf::thread::get_lane_id() == sn) {
					const auto norm = cutf::math::sqrt(norm2);
					const auto y_i = smem_y_ptr[gn];
					smem_y_ptr[gn] = y_i + cutf::math::sign(y_i) * norm;
				}
			}
			__syncthreads();
			MTK_DEBUG_PRINT_MATRIX(smem_y_ptr, 1, m, 1, "y (|y| added)");

			// Compute norm2 of y
			// TODO: Compute it from previous norm2
			const auto t = cutf::type::cast<T>(2.0f / compute_norm2<float>(smem_y_ptr, DIM_MAX_M));
			MTK_DEBUG_CALL_FUNC(printf("t = %e\n", t));
			
			// Compute ytA
			compute_reflection_0<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(smem_tmp_ptr, smem_y_ptr, smem_A_ptr, t);

			// Compute R
			compute_reflection_1<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(smem_A_ptr, smem_tmp_ptr, smem_y_ptr);

			smem_Y_ptr[sn * DIM_MAX_M + threadIdx.x] = smem_y_ptr[threadIdx.x];
			if (threadIdx.x == sn) {
				smem_t_ptr[sn] = t;
			}
		}
		MTK_CLOCK_BREAKDOWN_RECORD(2);
		MTK_DEBUG_PRINT_MATRIX(smem_Y_ptr, m, real_block_n, DIM_MAX_M, "Y (Block Result)");
		MTK_DEBUG_PRINT_MATRIX(smem_t_ptr, 1, real_block_n, 1, "t (Block Result)");
		// Store block Y and R
		mtk::tsqr_tc::utils::copy_matrix_s2g<block_size, DIM_BLOCK_N, DIM_MAX_M>(gmem_r_ptr + ldr * n_block * DIM_BLOCK_N, ldr, smem_A_ptr, n, real_block_n);
		mtk::tsqr_tc::utils::copy_matrix_s2g<block_size, DIM_BLOCK_N, DIM_MAX_M>(gmem_y_ptr + ldy * n_block * DIM_BLOCK_N, ldy, smem_Y_ptr, m, real_block_n);
		MTK_CLOCK_BREAKDOWN_RECORD(3);

		// Compute W
		__syncthreads();
		compute_base_w<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(
				smem_Y_ptr,
				smem_W_ptr,
				smem_A_ptr,
				smem_tmp_ptr,
				smem_t_ptr,
				gmem_w_ptr, ldw,
				gmem_y_ptr, ldy,
				m, n_block * DIM_BLOCK_N,
				real_block_n
				);
		MTK_CLOCK_BREAKDOWN_RECORD(4);
		MTK_DEBUG_PRINT_MATRIX(smem_W_ptr, m, real_block_n, DIM_MAX_M, "base W (Block Result)");
		compute_w<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(
				smem_W_ptr,
				smem_tmp_ptr,
				smem_Y_ptr,
				smem_t_ptr,
				m, real_block_n
				);
		mtk::tsqr_tc::utils::copy_matrix_s2g<block_size, DIM_BLOCK_N, DIM_MAX_M>(gmem_w_ptr + ldw * n_block * DIM_BLOCK_N, ldw, smem_W_ptr, m, real_block_n);
		MTK_DEBUG_PRINT_MATRIX(smem_W_ptr, m, real_block_n, DIM_MAX_M, "W (Block Result)");
		MTK_CLOCK_BREAKDOWN_RECORD(5);
#ifdef MTK_CLOCK_BREAKDOWN
		if (threadIdx.x + blockIdx.x == 0) {
			if (n_block == 0) {
				printf("n_block,update_a,local_householder,store,compute_base_w,compute_w\n");
			}
			printf("%lu,%lld,%lld,%lld,%lld,%lld\n",
				n_block,
				MTK_CLOCK_BREAKDOWN_DURATION(0, 1),
				MTK_CLOCK_BREAKDOWN_DURATION(1, 2),
				MTK_CLOCK_BREAKDOWN_DURATION(2, 3),
				MTK_CLOCK_BREAKDOWN_DURATION(3, 4),
				MTK_CLOCK_BREAKDOWN_DURATION(4, 5)
				);
		}
#endif
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
__global__ void qr256x128_kernel(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_r_ptr, const std::size_t ldr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n) {
	qr_kernel<compute_mode>(
			gmem_w_ptr, ldw,
			gmem_y_ptr, ldy,
			gmem_r_ptr, ldr,
			gmem_a_ptr, lda,
			m, n
			);
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
__global__ void qr256x128_batched_kernel(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_r_ptr, const std::size_t ldr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t n,
		const std::size_t batch_size,
		const std::size_t* const start_m_list) {
	const auto matrix_id = blockIdx.x;
	if (matrix_id >= batch_size) {
		return;
	}
	const auto start_m = (start_m_list == nullptr) ? (matrix_id * 2 * n)       : start_m_list[matrix_id];
	const auto end_m   = (start_m_list == nullptr) ? ((matrix_id + 1) * 2 * n) : start_m_list[matrix_id + 1];
	const auto m = end_m - start_m;
	qr_kernel<compute_mode>(
			gmem_w_ptr + start_m, ldw,
			gmem_y_ptr + start_m, ldy,
			gmem_r_ptr + matrix_id * n, ldr,
			gmem_a_ptr + start_m, lda,
			m, n
			);
}
} // noname namespace

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::qr256x128(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_r_ptr, const std::size_t ldr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n,
		const cudaStream_t cuda_stream
		) {
	const unsigned block_size = 256;
	const unsigned smem_size = 58432; //[B]
	cudaFuncSetAttribute(qr256x128_kernel<compute_mode>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
	qr256x128_kernel<compute_mode><<<1, block_size, smem_size, cuda_stream>>>(
			gmem_w_ptr, ldw,
			gmem_y_ptr, ldy,
			gmem_r_ptr, ldr,
			gmem_a_ptr, lda,
			m, n
			);
}

#define QR256X128_INSTANCE(compute_mode) \
template void mtk::tsqr_tc::qr256x128<compute_mode>( \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const std::size_t, \
		const std::size_t, \
		const cudaStream_t \
		)

QR256X128_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor   );
QR256X128_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor   );
QR256X128_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_no_cor);
QR256X128_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_no_cor);
QR256X128_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_no_tc           );

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::qr256x128_batched(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_r_ptr, const std::size_t ldr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t n,
		const std::size_t batch_size,
		const std::size_t* const start_m_list,
		const cudaStream_t cuda_stream
		) {
	const unsigned block_size = 256;
	const unsigned smem_size = 58432; //[B]
	cudaFuncSetAttribute(qr256x128_batched_kernel<compute_mode>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
	qr256x128_batched_kernel<compute_mode><<<batch_size, block_size, smem_size, cuda_stream>>>(
			gmem_w_ptr, ldw,
			gmem_y_ptr, ldy,
			gmem_r_ptr, ldr,
			gmem_a_ptr, lda,
			n,
			batch_size,
			start_m_list
			);
}

#define QR256X128_BATCHED_INSTANCE(compute_mode) \
template void mtk::tsqr_tc::qr256x128_batched<compute_mode>( \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const std::size_t, \
		const std::size_t, \
		const std::size_t* const, \
		const cudaStream_t \
		)

QR256X128_BATCHED_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor   );
QR256X128_BATCHED_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor   );
QR256X128_BATCHED_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_no_cor);
QR256X128_BATCHED_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_no_cor);
QR256X128_BATCHED_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_no_tc           );
