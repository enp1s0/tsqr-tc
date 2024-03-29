#include <cassert>
#include <tsqr_tc/tsqr.hpp>
#include <cutf/memory.hpp>
#include <cutf/thread.hpp>
#include <cutf/type.hpp>
#include <gemm_core/gemm_core.hpp>
#include "utils.hpp"
#include <tsqr_tc/detail/macro.hpp>

//#define MTK_DEBUG_DEVICE
//#define MTK_DEBUG_HOST
//#define MTK_CLOCK_BREAKDOWN

#ifdef MTK_DEBUG_DEVICE
#include <cutf/debug/matrix.hpp>
#include <type_traits>
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
//
#ifdef MTK_DEBUG_HOST
#include <cutf/debug/matrix.hpp>
#include <type_traits>
#define MTK_DEBUG_PRINT_DEVICE_MATRIX(ptr, m, n, ldm, name) \
{\
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());\
	auto h_matrix = cutf::memory::get_host_unique_ptr<std::remove_pointer<float>::type>(n * ldm); \
	cutf::memory::copy(h_matrix.get(), ptr, ldm * n); \
	cutf::debug::print::print_numpy_matrix(h_matrix.get(), m, n, ldm, name);\
}
#define MTK_DEBUG_CALL_HOST_FUNC(func) \
	func;std::fflush(stdout);
#define MTK_DEBUG_CHECK_KERNEL_ERROR \
	CUTF_CHECK_ERROR(cudaDeviceSynchronize())
#else
#define MTK_DEBUG_PRINT_DEVICE_MATRIX(ptr, m, n, ldm, name)
#define MTK_DEBUG_CALL_HOST_FUNC(func)
#define MTK_DEBUG_CHECK_KERNEL_ERROR
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
constexpr unsigned warp_size = 32;

template <unsigned block_size, unsigned DIM_N, unsigned MINUS, class DST_T, class SRC_T>
__device__ void copy_B_g2s(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr, const std::size_t ld_src,
		const unsigned n
		) {
	constexpr auto num_warps = block_size / warp_size;
	for (unsigned bm = 0; bm < DIM_N; bm += warp_size) {
		const auto real_bm = min(warp_size, n - bm);
		if (real_bm == warp_size) {
			for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				auto v = cutf::type::cast<SRC_T>(0.0f);
				if (gn < n) {
					v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
				}
				if constexpr (MINUS) {
					v = -v;
				}
				dst_ptr[gn * DIM_N + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
			}
		} else {
			for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				auto v = cutf::type::cast<SRC_T>(0.0f);
				if (gn < n && cutf::thread::get_lane_id() < real_bm) {
					v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
				}
				if constexpr (MINUS) {
					v = -v;
				}
				dst_ptr[gn * DIM_N + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
			}
		}
	}
}

template <unsigned block_size, unsigned DIM_XP, unsigned DIM_N, unsigned TRANS, class DST_T, class SRC_T>
__device__ void copy_matrix_g2s_XPxN(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr, const std::size_t ld_src,
		const unsigned m, const unsigned n
		) {
	constexpr auto num_warps = block_size / warp_size;
	MTK_DEBUG_CALL_FUNC(printf("# ->> %s<DIM_XP = %3u, DIM_N = %3u, TRANS = %u>(m = %3u, n = %3u, lda = %5lu)\n", __func__, DIM_XP, DIM_N, TRANS, m, n, ld_src));

	if constexpr (TRANS == 0) {
		for (unsigned bm = 0; bm < DIM_XP; bm += warp_size) {
			const auto real_bm = min(warp_size, max(m, bm) - bm);
			if (real_bm == warp_size) {
				for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
					const auto gn = bn + cutf::thread::get_warp_id();
					auto v = cutf::type::cast<SRC_T>(0.0f);
					if (gn < n) {
						v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
					}
					dst_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			} else {
				for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
					const auto gn = bn + cutf::thread::get_warp_id();
					auto v = cutf::type::cast<SRC_T>(0.0f);
					if (gn < n && cutf::thread::get_lane_id() < real_bm) {
						v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
					}
					dst_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		}
	} else {
		for (unsigned bm = 0; bm < DIM_XP; bm += warp_size) {
			const auto real_bm = min(warp_size, max(m, bm) - bm);
			if (real_bm == warp_size) {
				for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
					const auto gn = bn + cutf::thread::get_warp_id();
					auto v = cutf::type::cast<SRC_T>(0.0f);
					if (gn < n) {
						v = src_ptr[gn + (cutf::thread::get_lane_id() + bm) * ld_src];
					}
					dst_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			} else {
				for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
					const auto gn = bn + cutf::thread::get_warp_id();
					auto v = cutf::type::cast<SRC_T>(0.0f);
					if (gn < n && cutf::thread::get_lane_id() < real_bm) {
						v = src_ptr[gn + (cutf::thread::get_lane_id() + bm) * ld_src];
					}
					dst_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		}
	}
}

template <unsigned block_size, unsigned DIM_XP, unsigned DIM_N, class DST_T, class SRC_T>
__device__ void copy_matrix_s2g_XPxN(
		DST_T* const dst_ptr, const std::size_t ld_dst,
		const SRC_T* const src_ptr,
		const unsigned m, const unsigned n
		) {
	constexpr auto num_warps = block_size / warp_size;
	for (unsigned bm = 0; bm < m; bm += warp_size) {
		const auto real_bm = min(warp_size, m - bm);
		if (real_bm == warp_size) {
			for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				if (gn < n) {
					const auto v = src_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm];
					dst_ptr[gn * ld_dst + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		} else {
			for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				if (gn < n && cutf::thread::get_lane_id() < real_bm) {
					const auto v = src_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm];
					dst_ptr[gn * ld_dst + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		}
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, int A_TRANS, int B_MINUS, int C_EXIST>
__device__ void gemm_MxNxN_core_hmma(
		float* const gmem_D_ptr, const std::size_t ld_D,
		const float* const gmem_A_ptr, const std::size_t ld_A,
		const float* const gmem_B_ptr, const std::size_t ld_B,
		const float* const gmem_C_ptr, const std::size_t ld_C,
		const std::size_t m, const std::size_t n
		) {
	MTK_DEBUG_CALL_FUNC(printf("# ->> %s<A_TRANS = %d, B_MINUS = %d, C_EXIST = %d>(%3lu, %3lu)\n", __func__, A_TRANS, B_MINUS, C_EXIST, m, n));
	constexpr unsigned block_size = 256;
	constexpr std::size_t DIM_N = 128;
	constexpr std::size_t DIM_BLOCK_M = 64;
	constexpr std::size_t K_BLOCKING = 64;
	constexpr std::size_t DIM_TC = 16;
	constexpr std::size_t NUM_BLOCKINGS = K_BLOCKING / DIM_TC;

	extern __shared__ float smem[];
	float* const smem_B_ptr = smem;
	float* const smem_A_ptr = smem_B_ptr + DIM_N * DIM_N;

	// Load B
	copy_B_g2s<block_size, DIM_N, B_MINUS>(
			smem_B_ptr,
			gmem_B_ptr, ld_B,
			n
			);
	MTK_DEBUG_PRINT_MATRIX(smem_B_ptr, n, n, DIM_N, "B");
	__syncthreads();
	for (std::size_t bm = 0; bm < m; bm += DIM_BLOCK_M) {
		typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::accumulator, DIM_BLOCK_M / DIM_TC * DIM_TC, DIM_TC, NUM_BLOCKINGS * DIM_TC>::type frag_C;
		if constexpr (C_EXIST) {
			// We use n x n size of matrix C in TSQR even if the size of D is 2n x n.
			if (bm < n) {
				const auto real_m = min(DIM_BLOCK_M, n - bm);
				copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N, 0>(
						smem_A_ptr,
						gmem_C_ptr + bm, ld_C,
						real_m, n
						);
				MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, DIM_BLOCK_M, n, DIM_BLOCK_M, "C");
				__syncthreads();
				auto load_ptr = smem_A_ptr + DIM_BLOCK_M * DIM_TC * cutf::thread::get_warp_id();
				mtk::wmma::tcec::load_matrix_sync(frag_C, load_ptr, DIM_BLOCK_M, nvcuda::wmma::mem_col_major, false);
			} else {
				mtk::wmma::tcec::fill_zero(frag_C);
			}
		} else {
			MTK_TSQR_TC_UNUSED(gmem_C_ptr);
			MTK_TSQR_TC_UNUSED(ld_C);
			mtk::wmma::tcec::fill_zero(frag_C);
		}
		const auto real_m = min(DIM_BLOCK_M, m - bm);
		MTK_DEBUG_PRINT_MATRIX(gmem_A_ptr, m, n, ld_A, "GMEM_A");
		if constexpr (A_TRANS == 0) {
			copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N, A_TRANS>(
					smem_A_ptr,
					gmem_A_ptr + bm, ld_A,
					real_m, n
					);
		} else {
			copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N, A_TRANS>(
					smem_A_ptr,
					gmem_A_ptr + bm * ld_A, ld_A,
					real_m, n
					);
		}
		MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, real_m, n, DIM_BLOCK_M, "A block");
		__syncthreads();
		for (auto bk = decltype(DIM_N)(0); bk < DIM_N; bk += K_BLOCKING) {
			typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_b, DIM_BLOCK_M / DIM_TC * DIM_TC, DIM_TC, NUM_BLOCKINGS * DIM_TC, nvcuda::wmma::col_major>::type frag_B;
			mtk::wmma::tcec::load_matrix_sync(frag_B, smem_B_ptr + cutf::thread::get_warp_id() * DIM_N * DIM_TC + bk, DIM_N, false);

			typename mtk::tsqr_tc::utils::select_fragment<compute_mode, nvcuda::wmma::matrix_a, DIM_BLOCK_M / DIM_TC * DIM_TC, DIM_TC, NUM_BLOCKINGS * DIM_TC, nvcuda::wmma::col_major>::type frag_A;
			mtk::wmma::tcec::load_matrix_sync(frag_A, smem_A_ptr + bk * DIM_BLOCK_M, DIM_BLOCK_M, false);
			mtk::wmma::tcec::mma_sync(frag_C, frag_A, frag_B, frag_C);
		}
		__syncthreads();
		mtk::wmma::tcec::store_matrix_sync(smem_A_ptr + DIM_BLOCK_M * DIM_TC * cutf::thread::get_warp_id(), frag_C, DIM_BLOCK_M, nvcuda::wmma::mem_col_major, false);
		__syncthreads();
		copy_matrix_s2g_XPxN<block_size, DIM_BLOCK_M, DIM_N>(
				gmem_D_ptr + bm, ld_D,
				smem_A_ptr,
				real_m, n
				);
		MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, real_m, n, DIM_BLOCK_M, "D block");
	}
	MTK_DEBUG_CALL_FUNC(printf("# -<< %s<%d, %d, %d>\n", __func__, A_TRANS, B_MINUS, C_EXIST));
}

template <int A_TRANS, int B_MINUS, int C_EXIST>
__device__ void gemm_MxNxN_core_notc(
		float* const gmem_D_ptr, const std::size_t ld_D,
		const float* const gmem_A_ptr, const std::size_t ld_A,
		const float* const gmem_B_ptr, const std::size_t ld_B,
		const float* const gmem_C_ptr, const std::size_t ld_C,
		const std::size_t m, const std::size_t n
		) {
	MTK_DEBUG_CALL_FUNC(printf("# ->> %s<A_TRANS = %d, B_MINUS = %d, C_EXIST = %d>(%3lu, %3lu)\n", __func__, A_TRANS, B_MINUS, C_EXIST, m, n));
	constexpr unsigned block_size = 256;
	constexpr std::size_t DIM_N = 128;
	constexpr std::size_t DIM_BLOCK_M = 64;
	constexpr std::size_t DIM_TC = 16;

	extern __shared__ float smem[];
	float* const smem_B_ptr = smem;
	float* const smem_A_ptr = smem_B_ptr + DIM_N * DIM_N;

	// Load B
	copy_B_g2s<block_size, DIM_N, B_MINUS>(
			smem_B_ptr,
			gmem_B_ptr, ld_B,
			n
			);
	MTK_DEBUG_PRINT_MATRIX(smem_B_ptr, n, n, DIM_N, "B");
	MTK_DEBUG_PRINT_MATRIX(gmem_A_ptr, m, n, ld_A, "GMEM_A");
	__syncthreads();
	for (std::size_t bm = 0; bm < m; bm += DIM_BLOCK_M) {
		const auto real_m = min(DIM_BLOCK_M, m - bm);
		if constexpr (A_TRANS == 0) {
			copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N, A_TRANS>(
					smem_A_ptr,
					gmem_A_ptr + bm, ld_A,
					real_m, n
					);
		} else {
			copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N, A_TRANS>(
					smem_A_ptr,
					gmem_A_ptr + bm * ld_A, ld_A,
					real_m, n
					);
		}
		MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, real_m, n, DIM_BLOCK_M, "A block");
		__syncthreads();

		float reg[DIM_BLOCK_M * DIM_TC / warp_size];
		for (unsigned i = 0; i < DIM_BLOCK_M * DIM_TC / warp_size; i++) {
			reg[i] = 0.0f;
		}
		for (unsigned i = 0; i < DIM_TC; i++) {
			for (unsigned k = 0; k < n; k++) {
				const auto b_v = smem_B_ptr[cutf::thread::get_warp_id() * DIM_N * DIM_TC + k + i * DIM_N];
				for (unsigned j = 0; j < DIM_BLOCK_M; j += warp_size) {
					const auto a_v = smem_A_ptr[k * DIM_BLOCK_M + j + (threadIdx.x & 0x1f)];
					reg[i * (DIM_BLOCK_M / warp_size) + j / warp_size] += a_v * b_v;
				}
			}
		}
		__syncthreads();
		if constexpr (C_EXIST) {
			if (bm < n) {
				const auto real_m = min(DIM_BLOCK_M, n - bm);
				copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N, 0>(
						smem_A_ptr,
						gmem_C_ptr + bm, ld_C,
						real_m, n
						);
				MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, DIM_BLOCK_M, n, DIM_BLOCK_M, "C");
				__syncthreads();
				for (unsigned i = 0; i < DIM_TC; i++) {
					for (unsigned j = 0; j < DIM_BLOCK_M; j += warp_size) {
						smem_A_ptr[cutf::thread::get_warp_id() * DIM_BLOCK_M * DIM_TC + j + (threadIdx.x & 0x1f) + i * DIM_BLOCK_M] += reg[i * DIM_BLOCK_M / warp_size + j / warp_size];
					}
				}
			} else {
				for (unsigned i = 0; i < DIM_TC; i++) {
					for (unsigned j = 0; j < DIM_BLOCK_M; j += warp_size) {
						smem_A_ptr[cutf::thread::get_warp_id() * DIM_BLOCK_M * DIM_TC + j + (threadIdx.x & 0x1f) + i * DIM_BLOCK_M] = reg[i * DIM_BLOCK_M / warp_size + j / warp_size];
					}
				}
			}
		} else {
			MTK_TSQR_TC_UNUSED(gmem_C_ptr);
			MTK_TSQR_TC_UNUSED(ld_C);
			for (unsigned i = 0; i < DIM_TC; i++) {
				for (unsigned j = 0; j < DIM_BLOCK_M; j += warp_size) {
					smem_A_ptr[cutf::thread::get_warp_id() * DIM_BLOCK_M * DIM_TC + j + (threadIdx.x & 0x1f) + i * DIM_BLOCK_M] = reg[i * DIM_BLOCK_M / warp_size + j / warp_size];
				}
			}
		}
		__syncthreads();
		copy_matrix_s2g_XPxN<block_size, DIM_BLOCK_M, DIM_N>(
				gmem_D_ptr + bm, ld_D,
				smem_A_ptr,
				real_m, n
				);
		MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, real_m, n, DIM_BLOCK_M, "D block");
	}
	MTK_DEBUG_CALL_FUNC(printf("# -<< %s<%d, %d, %d>\n", __func__, A_MINUS, A_TRANS, C_EXIST));
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, int A_TRANS, int B_MINUS, int C_EXIST>
__device__ void gemm_MxNxN_core(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_D_ptr, const std::size_t ld_D,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_A_ptr, const std::size_t ld_A,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_B_ptr, const std::size_t ld_B,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_C_ptr, const std::size_t ld_C,
		const std::size_t m, const std::size_t n
		) {
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_no_tc) {
		gemm_MxNxN_core_notc<A_TRANS, B_MINUS, C_EXIST>(
				gmem_D_ptr, ld_D,
				gmem_A_ptr, ld_A,
				gmem_B_ptr, ld_B,
				gmem_C_ptr, ld_C,
				m, n
				);
	} else {
		gemm_MxNxN_core_hmma<compute_mode, A_TRANS, B_MINUS, C_EXIST>(
				gmem_D_ptr, ld_D,
				gmem_A_ptr, ld_A,
				gmem_B_ptr, ld_B,
				gmem_C_ptr, ld_C,
				m, n
				);
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
__global__ void tsqr_backward_kernel(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_OUT_Q_ptr  , const std::size_t ld_OUT_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_W_ptr      , const std::size_t ld_W,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_Y_ptr      , const std::size_t ld_Y,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_IN_Q_ptr   , const std::size_t ld_IN_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_WORKING_ptr,
		const std::size_t n,
		const std::size_t* const m_start_list
		) {
	const auto matrix_id = blockIdx.x;
	const auto start_m = (m_start_list == nullptr) ? matrix_id       * 2 * n : m_start_list[matrix_id];
	const auto end_m   = (m_start_list == nullptr) ? (matrix_id + 1) * 2 * n : m_start_list[matrix_id + 1];
	const auto m = end_m - start_m;

	auto g_Y_ptr = gmem_Y_ptr + start_m;
	auto g_W_ptr = gmem_W_ptr + start_m;
	auto g_OUT_Q_ptr = gmem_OUT_Q_ptr + start_m;
	auto g_IN_Q_ptr = gmem_IN_Q_ptr + matrix_id * n;
	auto working_memory_ptr = gmem_WORKING_ptr + matrix_id * n * n;

	MTK_CLOCK_BREAKDOWN_INIT(3);

	MTK_CLOCK_BREAKDOWN_RECORD(0);
	gemm_MxNxN_core<compute_mode, 1, 0, 0>(
			working_memory_ptr, n,
			g_Y_ptr, ld_Y,
			g_IN_Q_ptr, ld_IN_Q,
			nullptr, 0,
			n, n
			);
	MTK_CLOCK_BREAKDOWN_RECORD(1);
	__syncthreads();
	gemm_MxNxN_core<compute_mode, 0, 1, 1>(
			g_OUT_Q_ptr, ld_OUT_Q,
			g_W_ptr, ld_W,
			working_memory_ptr, n,
			g_IN_Q_ptr, ld_IN_Q,
			m, n
			);
	MTK_CLOCK_BREAKDOWN_RECORD(2);
#ifdef MTK_CLOCK_BREAKDOWN
	if (threadIdx.x + blockIdx.x == 0) {
		printf("batch_size,n,gemm1,gemm2\n");
		printf("%u,%lu,%lld,%lld\n", blockDim.x, n,
				MTK_CLOCK_BREAKDOWN_DURATION(0, 1),
				MTK_CLOCK_BREAKDOWN_DURATION(1, 2));
	}
#endif
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void tsqr_backward(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_OUT_Q_ptr  , const std::size_t ld_OUT_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_W_ptr      , const std::size_t ld_W,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_Y_ptr      , const std::size_t ld_Y,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_IN_Q_ptr   , const std::size_t ld_IN_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_WORKING_ptr,
		const std::size_t n,
		const std::size_t split_size,
		const std::size_t* const m_start_list,
		const cudaStream_t cuda_stream
		) {
	constexpr auto block_size = 256;
	constexpr auto shared_memory_size = 96 * 1024;

	cudaFuncSetAttribute(&tsqr_backward_kernel<compute_mode>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
	tsqr_backward_kernel<compute_mode><<<split_size, block_size, shared_memory_size, cuda_stream>>>(
			gmem_OUT_Q_ptr  , ld_OUT_Q,
			gmem_W_ptr      , ld_W,
			gmem_Y_ptr      , ld_Y,
			gmem_IN_Q_ptr   , ld_IN_Q,
			gmem_WORKING_ptr,
			n, m_start_list
			);
}

template <class T>
__global__ void make_identity_matrix_kernel(
		T* const ptr,
		unsigned m, unsigned n
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (m * n <= tid) {
		return;
	}

	T v = cutf::type::cast<T>(0.0f);
	if (tid / m == tid % m) {
		v = cutf::type::cast<T>(1.0f);
	}
	ptr[tid] = v;
}

template <class T>
void make_indentity_matrix(
		T* const ptr,
		const unsigned m, const unsigned n,
		const cudaStream_t cuda_stream
		) {
	const auto block_size = 256;

	make_identity_matrix_kernel<<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
			ptr,
			m, n
			);
}

} // noname namespace

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::tsqr(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const q_ptr, const std::size_t ld_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const r_ptr, const std::size_t ld_R,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const a_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		mtk::tsqr_tc::tsqr_buffer<compute_mode>& buffer,
		const cudaStream_t cuda_stream
		) {
	assert(n <= 128);
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("=======DEBUG START========\n= %s\n==========================\n", __func__));
	MTK_DEBUG_PRINT_DEVICE_MATRIX(a_ptr, m, n, ld_A, "A(Input)");
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("Split count : %lu\n", buffer.get_split_count()));
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("=======PREPERATION==\n"));

	MTK_DEBUG_CALL_HOST_FUNC([&](){std::printf("m_list = [");for (std::size_t i = 0; i < buffer.get_split_count() + 1; i++) {std::printf("%lu ", buffer.get_index_buffer_host_ptr()[i]);}std::printf("]\n");}());

	unsigned r_ptr_index = 0;
	typename mtk::tsqr_tc::tsqr_buffer<compute_mode>::buffer_type* r_buffer_list[2];
	r_buffer_list[0] = buffer.get_r_buffer_ptr();
	r_buffer_list[1] = buffer.get_r_buffer_ptr() + buffer.get_split_count() * n * n;

	std::size_t wy_ptr_offset = 0;

	// Forwad computation
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("=======FORWARD======\n"));
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("BQR (first) [wy_offset = %10lu, r_ptr_flipflop = %u]\n", wy_ptr_offset, r_ptr_index));
	mtk::tsqr_tc::qr256x128_batched<compute_mode>(
			buffer.get_w_buffer_ptr() + wy_ptr_offset, m,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, m,
			r_buffer_list[r_ptr_index], n * buffer.get_split_count(),
			a_ptr, ld_A,
			n,
			buffer.get_split_count(),
			buffer.get_index_buffer_ptr(),
			cuda_stream
			);
	MTK_DEBUG_PRINT_DEVICE_MATRIX(r_buffer_list[r_ptr_index], n * buffer.get_split_count(), n, n * buffer.get_split_count(), "Rs");
	wy_ptr_offset += m * n;
	r_ptr_index = 1 - r_ptr_index;

	for (std::size_t s = buffer.get_split_count() >> 1; s > 1; s >>= 1) {
		MTK_DEBUG_CALL_HOST_FUNC(std::printf("BQR (s=%3lu) [wy_offset = %10lu, r_ptr_flipflop = %u]\n", s, wy_ptr_offset, r_ptr_index));
		MTK_DEBUG_PRINT_DEVICE_MATRIX(r_buffer_list[1 - r_ptr_index], 2 * n * s, n, 2 * n * s, "Input of BQR");
		mtk::tsqr_tc::qr256x128_batched<compute_mode>(
				buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				r_buffer_list[r_ptr_index], n * s,
				r_buffer_list[1 - r_ptr_index], 2 * n * s,
				n,
				s,
				nullptr,
				cuda_stream
				);
		MTK_DEBUG_PRINT_DEVICE_MATRIX(r_buffer_list[r_ptr_index], 2 * n * s, n, 2 * n * s, ("Rs(s=" + std::to_string(s) + ", ldr=" + std::to_string(2 * n * s) + ")").c_str());
		wy_ptr_offset += 2 * n * n * s;
		r_ptr_index = 1 - r_ptr_index;
	}

	MTK_DEBUG_CALL_HOST_FUNC(std::printf("BQR (last ) [wy_offset = %10lu, r_ptr_flipflop = %u]\n", wy_ptr_offset, r_ptr_index));
	MTK_DEBUG_PRINT_DEVICE_MATRIX(r_buffer_list[1 - r_ptr_index], 2 * n, n, 2 * n, "Input of BQR");
	mtk::tsqr_tc::qr256x128<compute_mode>(
			buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n,
			r_ptr, ld_R,
			r_buffer_list[1 - r_ptr_index], 2 * n,
			2 * n, n,
			cuda_stream
			);
	MTK_DEBUG_PRINT_DEVICE_MATRIX(r_ptr, n, n, ld_R, "R");

	MTK_DEBUG_CALL_HOST_FUNC(std::printf("=======BACKWARD=====\n"));
	// Backward computation
	make_indentity_matrix(
			buffer.get_r_buffer_ptr(),
			n, n,
			cuda_stream
			);
	MTK_DEBUG_CHECK_KERNEL_ERROR;
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("BGEMM (last ) [wy_offset = %10lu]\n", wy_ptr_offset));
	MTK_DEBUG_PRINT_DEVICE_MATRIX(buffer.get_r_buffer_ptr(), n, n, n, "Input Q (Identity)");
	tsqr_backward<compute_mode>(
			buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_r_buffer_ptr(), n,
			buffer.get_r_buffer_ptr() + n * n,
			n, 1,
			nullptr,
			cuda_stream
			);
	MTK_DEBUG_CHECK_KERNEL_ERROR;
	MTK_DEBUG_PRINT_DEVICE_MATRIX(buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n, n, 2 * n, "First Q");
	for (std::size_t s = 2; s < buffer.get_split_count(); s <<= 1) {
		const auto prev_wy_ptr_offset = wy_ptr_offset;
		wy_ptr_offset -= 2 * n * n * s;
		MTK_DEBUG_CALL_HOST_FUNC(std::printf("BGEMM (s= %3lu) [wy_offset = %10lu]\n", s, wy_ptr_offset));
		tsqr_backward<compute_mode>(
				buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_w_buffer_ptr() + prev_wy_ptr_offset, (2 * n * s) / 2,
				buffer.get_r_buffer_ptr(),
				n, s,
				nullptr,
				cuda_stream
				);
		MTK_DEBUG_CHECK_KERNEL_ERROR;
		MTK_DEBUG_PRINT_DEVICE_MATRIX(buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s, n, 2 * n * s, ("Middle Q(s=" + std::to_string(s) + ")").c_str());
	}
	const auto prev_wy_ptr_offset = wy_ptr_offset;
	wy_ptr_offset -= m * n;
	const auto s = buffer.get_split_count();
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("BGEMM (first) [wy_offset = %10lu]\n", wy_ptr_offset));
	tsqr_backward<compute_mode>(
			q_ptr, ld_Q,
			buffer.get_w_buffer_ptr() + wy_ptr_offset, m,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, m,
			buffer.get_w_buffer_ptr() + prev_wy_ptr_offset, (2 * n * s) / 2,
			buffer.get_r_buffer_ptr(),
			n, s,
			buffer.get_index_buffer_ptr(),
			cuda_stream
			);
	MTK_DEBUG_CHECK_KERNEL_ERROR;
	MTK_DEBUG_CALL_HOST_FUNC(std::printf("=======DEBUG END  ========\n= %s\n==========================\n", __func__));
}

#define MTK_INSTANCE_TSQR(compute_mode) \
template void mtk::tsqr_tc::tsqr(\
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const std::size_t, const std::size_t, \
		mtk::tsqr_tc::tsqr_buffer<compute_mode>&, \
		const cudaStream_t\
		)

MTK_INSTANCE_TSQR(mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor   );
MTK_INSTANCE_TSQR(mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor   );
MTK_INSTANCE_TSQR(mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_no_cor);
MTK_INSTANCE_TSQR(mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_no_cor);
MTK_INSTANCE_TSQR(mtk::tsqr_tc::compute_mode::fp32_no_tc           );
