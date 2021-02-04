#include <cstdint>
#include <mma.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include <cutf/thread.hpp>
#include <wmma_extension.hpp>

#include <tsqr_tc/batchedqr.hpp>

//#define MTK_DEBUG
#ifdef MTK_DEBUG
#include <cutf/debug/matrix.hpp>
#define MTK_DEBUG_PRINT_MATRIX(ptr, m, n, ldm, name) \
	__syncthreads(); \
	if (threadIdx.x == 0) cutf::debug::print::print_numpy_matrix(ptr, m, n, ldm, name); \
	__syncthreads();
#define MTK_DEBUG_CALL_FUNC(func) \
	__syncthreads(); \
	if (threadIdx.x == 0) func; \
	__syncthreads();
#else
#define MTK_DEBUG_PRINT_MATRIX(ptr, m, n, ldm, name)
#define MTK_DEBUG_CALL_FUNC(func);
#endif

namespace {

constexpr unsigned warp_size = 32u;

// This function fills memory with zero
template <unsigned block_size, unsigned size, class T>
__device__ void fill_zero(T* const ptr) {
	if constexpr (size % block_size == 0) {
		for (unsigned i = 0; i < size; i += block_size) {
			const auto index = i + threadIdx.x;
			ptr[index] = cutf::type::cast<T>(0.0f);
		}
	} else {
		for (unsigned index = threadIdx.x; index < size; index += block_size) {
			ptr[index] = cutf::type::cast<T>(0.0f);
		}
	}
}

// This function copies matrix data from global memory to shared memory
// Ristrictions:
// - smem_m == block_size
template <unsigned block_size, unsigned smem_n, unsigned smem_ld, class SMEM_T, class GMEM_T>
__device__ void copy_matrix_g2s(
		SMEM_T* const smem,
		const GMEM_T* const gmem_ptr, const std::size_t gmem_ld,
		const std::size_t m, const std::size_t n
		) {
	if (m == block_size) {
		unsigned i_n = 0;
		for (; i_n < n; i_n++) {
			const auto v = gmem_ptr[gmem_ld * i_n + threadIdx.x];
			smem[smem_ld * i_n + threadIdx.x] = cutf::type::cast<SMEM_T>(v);
		}
		for (; i_n < smem_n; i_n++) {
			smem[smem_ld * i_n + threadIdx.x] = cutf::type::cast<SMEM_T>(0.0f);
		}
	} else {
		unsigned i_n = 0;
		for (; i_n < n; i_n++) {
			auto v = cutf::type::cast<GMEM_T>(0.0f);
			if (threadIdx.x < m) {
				v = gmem_ptr[gmem_ld * i_n + threadIdx.x];
			}
			smem[smem_ld * i_n + threadIdx.x] = cutf::type::cast<SMEM_T>(v);
		}
		for (; i_n < smem_n; i_n++) {
			smem[smem_ld * i_n + threadIdx.x] = cutf::type::cast<SMEM_T>(0.0f);
		}
	}
}

// This function copies matrix data from shared memory to global memory
// Ristrictions:
// - smem_m == block_size
template <unsigned block_size, unsigned smem_n, unsigned smem_ld, class SMEM_T, class GMEM_T>
__device__ void copy_matrix_s2g(
		GMEM_T* const gmem_ptr, const std::size_t gmem_ld,
		const SMEM_T* const smem,
		const std::size_t m, const std::size_t n
		) {
	if (m == block_size) {
		unsigned i_n = 0;
		for (; i_n < n; i_n++) {
			const auto v = smem[smem_ld * i_n + threadIdx.x];
			gmem_ptr[gmem_ld * i_n + threadIdx.x] = cutf::type::cast<GMEM_T>(v);
		}
	} else {
		if (threadIdx.x < m) {
			for (unsigned i_n = 0; i_n < n; i_n++) {
				const auto v = smem[smem_ld * i_n + threadIdx.x];
				gmem_ptr[gmem_ld * i_n + threadIdx.x] = cutf::type::cast<GMEM_T>(v);
			}
		}
	}
}

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

// This function accumulates vectors on shared memory.
// Restrictions:
// count == block_size / war_size
// output_ptr == inpute_ptr
template <unsigned block_size, class T>
__device__ void accumulate_vectors(T* const smem_vec_ptr, const unsigned vec_len) {
	for (unsigned whole_vec_len = vec_len * block_size / warp_size; whole_vec_len > vec_len; whole_vec_len >>= 1) {
		for (unsigned offset = 0; offset < whole_vec_len / 2; offset += block_size) {
			const auto index = offset + threadIdx.x;
			if (index > vec_len) break;

			smem_vec_ptr[index] += smem_vec_ptr[index + whole_vec_len / 2];
		}
		__syncthreads();
	}
}

// This function computes `tmp = y^t * A`.
// Restrictions:
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_0_fp32_hmma_cor(
		float* const smem_reduction,
		const float* const smem_y,
		const float* const smem_A
		) {
	constexpr unsigned num_accumulate = warp_size / smem_n;
	constexpr float cor_scale = 1024.0f;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, smem_n, smem_n, smem_n, half, nvcuda::wmma::row_major> frag_yt[num_accumulate], frag_d_yt[num_accumulate];
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, smem_n, smem_n, smem_n, half, nvcuda::wmma::col_major> frag_a[num_accumulate], frag_d_a[num_accumulate];
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, smem_n, smem_n, smem_n, float> frag_ytA, frag_d_ytA;
	mtk::wmma::fill_zero(frag_ytA);
	mtk::wmma::fill_zero(frag_d_ytA);

	// Load A
	mtk::wmma::foreach<decltype(frag_a[0])>([&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
				const auto offset = (mem_index / smem_n) * smem_ldm;
				const auto row = mem_index % smem_n + (threadIdx.x & 0xffffffe0u);
				for (unsigned k = 0; k < num_accumulate; k++) {
					const auto r = row + k * smem_n;
					const auto v = smem_A[offset + r];
					const auto hv = cutf::type::cast<half>(v);
					const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
					for (unsigned i = 0; i < frag_index_count; i++) {
					const unsigned frag_index = frag_index_list[i];
						frag_a[k].x[frag_index] = hv;
						frag_d_a[k].x[frag_index] = dhv;
					}
				}
			});
	mtk::wmma::foreach_v<decltype(frag_yt[0])>(
			[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
				for (unsigned k = 0; k < num_accumulate; k++) {
					const auto row = k * smem_n + mem_index + (threadIdx.x & 0xffffffe0u);
					const auto v = smem_y[row];
					const auto hv = cutf::type::cast<half>(v);
					const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
					for (unsigned i = 0; i < fragment_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						frag_yt[k].x[frag_index] = hv;
						frag_d_yt[k].x[frag_index] = dhv;
					}
				}
			});

	for (unsigned k = 0; k < num_accumulate; k++) {
		// Compute (y^t * A)
		nvcuda::wmma::mma_sync(frag_ytA  , frag_yt[k]  , frag_a[k]  , frag_ytA  );
		nvcuda::wmma::mma_sync(frag_d_ytA, frag_yt[k]  , frag_d_a[k], frag_d_ytA);
		nvcuda::wmma::mma_sync(frag_d_ytA, frag_d_yt[k], frag_a[k]  , frag_d_ytA);
	}

	// Store
	mtk::wmma::foreach_v<decltype(frag_ytA)>(nvcuda::wmma::mem_row_major,
			[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
				float* res_ptr = smem_reduction + smem_n * (threadIdx.x >> 5);
				for (unsigned i = 0; i < fragment_index_count; i++) {
					const auto frag_index = frag_index_list[i];
					res_ptr[mem_index] = frag_ytA.x[frag_index] + frag_d_ytA.x[frag_index] / cor_scale;
				}
			});

	// Accumulate
	__syncthreads();
	accumulate_vectors<smem_m>(smem_reduction, smem_n);
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_0(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_reduction_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_y_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_a_ptr
		) {
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_hmma_cor) {
		compute_reflection_0_fp32_hmma_cor<smem_m, smem_n, smem_ldm>(smem_reduction_ptr, smem_y_ptr, smem_a_ptr);
	}
}

// This function computes `A = A -2t * y * tmp`.
// Restrictions:
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_1_fp32_hmma_cor(
		float* const smem_A_ptr,
		float* const smem_reduction_ptr,
		const float* const smem_y_ptr,
		const float t
		) {
	constexpr unsigned num_col_block = warp_size / smem_n;

	if (threadIdx.x < smem_n) {
		smem_reduction_ptr[threadIdx.x] *= -t;
	}

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, smem_n, smem_n, smem_n, half, nvcuda::wmma::col_major> frag_y;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, smem_n, smem_n, smem_n, half, nvcuda::wmma::row_major> frag_tmp;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, smem_n, smem_n, smem_n, float> frag_A;

	mtk::wmma::fill_zero(frag_y);
	__syncthreads();
	mtk::wmma::make_direct_product_fragment(frag_tmp, smem_reduction_ptr);

	for (unsigned i = 0; i < num_col_block; i++) {
		mtk::wmma::make_direct_product_fragment(frag_y, smem_y_ptr + i * smem_n + (threadIdx.x & 0xffffffe0u));
		nvcuda::wmma::load_matrix_sync(frag_A, smem_A_ptr + i * smem_n + (threadIdx.x & 0xffffffe0u), smem_ldm, nvcuda::wmma::mem_col_major);

		nvcuda::wmma::mma_sync(frag_A, frag_y, frag_tmp, frag_A);

		nvcuda::wmma::store_matrix_sync(smem_A_ptr + i * smem_n + (threadIdx.x & 0xffffffe0u), frag_A, smem_ldm, nvcuda::wmma::mem_col_major);
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_reflection_1(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_A_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_reduction_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_y_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type t
		) {
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_hmma_cor) {
		compute_reflection_1_fp32_hmma_cor<smem_m, smem_n, smem_ldm>(smem_A_ptr, smem_reduction_ptr, smem_y_ptr, t);
	}
}

// This function computes `w = (I - W * Y^t)y`.
// Restrictions:
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_w_fp32_hmma_cor(
		float* const smem_w_ptr,
		float* const smem_reduction_ptr,
		const float* const smem_y_ptr,
		const float* const smem_Y_ptr,
		const float* const smem_W_ptr,
		const float t
		) {
	constexpr unsigned num_col_block = warp_size / smem_n;
	const float cor_scale = 1024.0f;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, smem_n, smem_n, smem_n, half, nvcuda::wmma::col_major> frag_y[num_col_block], frag_d_y[num_col_block];
	for (unsigned i = 0; i < num_col_block; i++) {
		mtk::wmma::fill_zero(frag_y[i]);
		mtk::wmma::fill_zero(frag_d_y[i]);
	}

	{
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, smem_n, smem_n, smem_n, half, nvcuda::wmma::row_major> frag_Yt[num_col_block], frag_d_Yt[num_col_block];
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, smem_n, smem_n, smem_n, float> frag_tmp, frag_d_tmp;

		// Load Yt
		mtk::wmma::foreach<decltype(frag_Yt[0])>([&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					const auto offset = (mem_index / smem_n) * smem_ldm;
					const auto col = mem_index % smem_n + (threadIdx.x & 0xffffffe0u);
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto c = col + k * smem_n;
						const auto v = smem_Y_ptr[offset + c];
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < frag_index_count; i++) {
							const unsigned frag_index = frag_index_list[i];
							frag_Yt[k].x[frag_index] = hv;
							frag_d_Yt[k].x[frag_index] = dhv;
						}
					}
				});
		// Load y
		mtk::wmma::foreach_v<decltype(frag_y[0])>(
				[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto row = k * smem_n + mem_index + (threadIdx.x & 0xffffffe0u);
						const auto v = smem_y_ptr[row];
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < fragment_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag_y[k].x[frag_index] = hv;
							frag_d_y[k].x[frag_index] = dhv;
						}
					}
				});

		mtk::wmma::fill_zero(frag_tmp);
		mtk::wmma::fill_zero(frag_d_tmp);
		for (unsigned k = 0; k < num_col_block; k++) {
			// Compute (Yt * y)
			nvcuda::wmma::mma_sync(frag_tmp  , frag_Yt[k]  , frag_y[k], frag_tmp  );
			nvcuda::wmma::mma_sync(frag_d_tmp, frag_d_Yt[k], frag_y[k], frag_d_tmp);
			nvcuda::wmma::mma_sync(frag_d_tmp, frag_Yt[k], frag_d_y[k], frag_d_tmp);
		}

		// Store
		mtk::wmma::foreach_v<decltype(frag_tmp)>(nvcuda::wmma::mem_col_major,
				[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
					float* res_ptr = smem_reduction_ptr + smem_n * (threadIdx.x >> 5);
					for (unsigned i = 0; i < fragment_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						res_ptr[mem_index] = frag_tmp.x[frag_index] + frag_d_tmp.x[frag_index] / cor_scale;
					}
				});
	}

	// Accumulate
	MTK_DEBUG_PRINT_MATRIX(smem_reduction_ptr, 1, smem_n, 1, "Yty (before accumulating)");
	__syncthreads();
	accumulate_vectors<smem_m>(smem_reduction_ptr, smem_n);
	MTK_DEBUG_PRINT_MATRIX(smem_reduction_ptr, 1, smem_n, 1, "Yty (accumulated)");

	// Compute w <- W * tmp
	{
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, smem_n, smem_n, smem_n, half, nvcuda::wmma::col_major> frag_W[num_col_block], frag_d_W[num_col_block];
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, smem_n, smem_n, smem_n, float> frag_w[num_col_block], frag_d_w[num_col_block];
		// Load tmp
		mtk::wmma::foreach_v<decltype(frag_y[0])>(
				[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
					const auto v = smem_reduction_ptr[mem_index];
					const auto hv = cutf::type::cast<half>(v);
					const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
					for (unsigned i = 0; i < fragment_index_count; i++) {
						const auto frag_index = frag_index_list[i];
						frag_y[0].x[frag_index] = hv;
						frag_d_y[0].x[frag_index] = dhv;
					}
				});
		// Load W
		mtk::wmma::foreach<decltype(frag_W[0])>([&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					const auto offset = (mem_index / smem_n) * smem_ldm;
					const auto row = mem_index % smem_n + (threadIdx.x & 0xffffffe0u);
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto r = row + k * smem_n;
						const auto v = smem_W_ptr[offset + r];
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < frag_index_count; i++) {
						const unsigned frag_index = frag_index_list[i];
							frag_W[k].x[frag_index] = hv;
							frag_d_W[k].x[frag_index] = dhv;
						}
					}
				});
		for (unsigned k = 0; k < num_col_block; k++) {
			mtk::wmma::fill_zero(frag_w[k]);
			mtk::wmma::fill_zero(frag_d_w[k]);
			// Compute (Yt * A)
			nvcuda::wmma::mma_sync(frag_w[k]  , frag_W[k]  , frag_y[0]  , frag_w[k]  );
			nvcuda::wmma::mma_sync(frag_d_w[k], frag_d_W[k], frag_y[0]  , frag_d_w[k]);
			nvcuda::wmma::mma_sync(frag_d_w[k], frag_W[k]  , frag_d_y[0], frag_d_w[k]);
		}
		// Store
		mtk::wmma::foreach_v<decltype(frag_w[0])>(nvcuda::wmma::mem_col_major,
				[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto offset = (threadIdx.x & 0xffffffe0u) + k * smem_n;
						float* const res_ptr = smem_w_ptr + offset;
						const float* const y_ptr = smem_y_ptr + offset;
						for (unsigned i = 0; i < fragment_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							res_ptr[mem_index] = (y_ptr[mem_index] - (frag_w[k].x[frag_index] + frag_d_w[k].x[frag_index] / cor_scale)) * t;
						}
					}
				});
		MTK_DEBUG_PRINT_MATRIX(smem_w_ptr, 1, smem_n, 1, "w");
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void compute_w(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_w_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_reduction_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_y_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_Y_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_W_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type t
		) {
	MTK_DEBUG_CALL_FUNC(printf("# --> %s\n", __func__));
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_hmma_cor) {
		compute_w_fp32_hmma_cor<smem_m, smem_n, smem_ldm>(smem_w_ptr, smem_reduction_ptr, smem_y_ptr, smem_Y_ptr, smem_W_ptr, t);
	}
	MTK_DEBUG_CALL_FUNC(printf("# <-- %s\n", __func__));
}

// This function computes `A <- (I - W * Y^T)A`.
// Restrictions:
// - smem_m == block_size
// - smem_n == DIM_BLOCK_N
template <unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void update_a_fp32_hmma_cor(
		float* const smem_A_ptr,
		float* const smem_YtA_ptr,
		const float* const smem_W_ptr,
		const float* const smem_Y_ptr
		) {
	constexpr unsigned num_col_block = warp_size / smem_n;
	const float cor_scale = 1024.0f;

	{
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, smem_n, smem_n, smem_n, half, nvcuda::wmma::row_major> frag_Yt[num_col_block], frag_d_Yt[num_col_block];
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, smem_n, smem_n, smem_n, half, nvcuda::wmma::col_major> frag_A[num_col_block], frag_d_A[num_col_block];
		// Load Yt
		mtk::wmma::foreach<decltype(frag_Yt[0])>([&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					const auto offset = (mem_index / smem_n) * smem_ldm;
					const auto row = mem_index % smem_n + (threadIdx.x & 0xffffffe0u);
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto r = row + k * smem_n;
						const auto v = smem_Y_ptr[offset + r];
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < frag_index_count; i++) {
						const unsigned frag_index = frag_index_list[i];
							frag_Yt[k].x[frag_index] = hv;
							frag_d_Yt[k].x[frag_index] = dhv;
						}
					}
				});
		// Load A
		mtk::wmma::foreach<decltype(frag_A[0])>([&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					const auto offset = (mem_index / smem_n) * smem_ldm;
					const auto row = mem_index % smem_n + (threadIdx.x & 0xffffffe0u);
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto r = row + k * smem_n;
						const auto v = smem_A_ptr[offset + r];
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < frag_index_count; i++) {
						const unsigned frag_index = frag_index_list[i];
							frag_A[k].x[frag_index] = hv;
							frag_d_A[k].x[frag_index] = dhv;
						}
					}
				});

		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, smem_n, smem_n, smem_n, float> frag_tmp, frag_d_tmp;
		mtk::wmma::fill_zero(frag_tmp);
		mtk::wmma::fill_zero(frag_d_tmp);
		for (unsigned k = 0; k < num_col_block; k++) {
			// Compute (Yt * A)
			nvcuda::wmma::mma_sync(frag_tmp  , frag_Yt[k]  , frag_A[k]  , frag_tmp  );
			nvcuda::wmma::mma_sync(frag_d_tmp, frag_d_Yt[k], frag_A[k]  , frag_d_tmp);
			nvcuda::wmma::mma_sync(frag_d_tmp, frag_Yt[k]  , frag_d_A[k], frag_d_tmp);
		}

		for (unsigned i = 0; i < frag_tmp.num_elements; i++) {
			frag_tmp.x[i] += frag_d_tmp.x[i] / cor_scale;
		}

		nvcuda::wmma::store_matrix_sync(smem_YtA_ptr + smem_n * smem_n * (threadIdx.x >> 5), frag_tmp, smem_n, nvcuda::wmma::mem_col_major);
	}

	__syncthreads();
	accumulate_vectors<smem_m>(smem_YtA_ptr, smem_n * smem_n);
	if (threadIdx.x < smem_n * smem_n) {
		smem_YtA_ptr[threadIdx.x] *= -1.0f;
	}
	__syncthreads();

	// Compute (A = A - W * YtA)
	{
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, smem_n, smem_n, smem_n, half, nvcuda::wmma::col_major> frag_W[num_col_block], frag_d_W[num_col_block];
		nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, smem_n, smem_n, smem_n, half, nvcuda::wmma::col_major> frag_YtA, frag_d_YtA;

		// Load Yt
		mtk::wmma::foreach<decltype(frag_W[0])>([&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					const auto offset = (mem_index / smem_n) * smem_ldm;
					const auto row = mem_index % smem_n + (threadIdx.x & 0xffffffe0u);
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto r = row + k * smem_n;
						const auto v = smem_W_ptr[offset + r];
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < frag_index_count; i++) {
						const unsigned frag_index = frag_index_list[i];
							frag_W[k].x[frag_index] = hv;
							frag_d_W[k].x[frag_index] = dhv;
						}
					}
				});
		// Load A
		mtk::wmma::foreach<decltype(frag_YtA)>([&](const unsigned frag_index_list[], const unsigned frag_index_count, const unsigned mem_index) {
					for (unsigned k = 0; k < num_col_block; k++) {
						const auto v = smem_YtA_ptr[mem_index];
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < frag_index_count; i++) {
						const unsigned frag_index = frag_index_list[i];
							frag_YtA.x[frag_index] = hv;
							frag_d_YtA.x[frag_index] = dhv;
						}
					}
				});

		for (unsigned k = 0; k < num_col_block; k++) {
			nvcuda::wmma::fragment<nvcuda::wmma::accumulator, smem_n, smem_n, smem_n, float> frag_A, frag_d_A;
			nvcuda::wmma::load_matrix_sync(frag_A, smem_A_ptr + (threadIdx.x & 0xffffffe0u), smem_ldm, nvcuda::wmma::mem_col_major);
			mtk::wmma::fill_zero(frag_d_A);
			// Compute (Yt * A)
			nvcuda::wmma::mma_sync(frag_A  , frag_W[k]  , frag_YtA  , frag_A  );
			nvcuda::wmma::mma_sync(frag_d_A, frag_d_W[k], frag_YtA  , frag_d_A);
			nvcuda::wmma::mma_sync(frag_d_A, frag_W[k]  , frag_d_YtA, frag_d_A);

			for (unsigned i = 0; i < frag_A.num_elements; i++) {
				frag_A.x[i] += frag_d_A.x[i] / cor_scale;
			}

			nvcuda::wmma::store_matrix_sync(smem_A_ptr + (threadIdx.x & 0xffffffe0u), frag_A, smem_n, nvcuda::wmma::mem_col_major);
		}
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, unsigned smem_m, unsigned smem_n, unsigned smem_ldm>
__device__ void update_a(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_A_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_YtA_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_W_ptr,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const smem_Y_ptr
		) {
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_hmma_cor) {
		update_a_fp32_hmma_cor<smem_m, smem_n, smem_ldm>(smem_A_ptr, smem_YtA_ptr, smem_W_ptr, smem_Y_ptr);
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
__device__ void qr_kernel(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
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
	T* const smem_tmp_ptr = smem_y_ptr + DIM_MAX_M;

	const unsigned num_n_blocks = (n + DIM_BLOCK_N - 1) / DIM_BLOCK_N;
	for (std::size_t n_block = 0; n_block < num_n_blocks; n_block++) {
		fill_zero<block_size, DIM_MAX_M * DIM_BLOCK_N>(smem_W_ptr);
		fill_zero<block_size, DIM_MAX_M * DIM_BLOCK_N>(smem_Y_ptr);

		const unsigned real_block_n = umin(DIM_BLOCK_N, n - DIM_BLOCK_N * n_block);
		copy_matrix_g2s<block_size, DIM_BLOCK_N, DIM_MAX_M>(smem_A_ptr, gmem_a_ptr + lda * n_block * DIM_BLOCK_N, lda, m, real_block_n);


		for (unsigned sn = 0; sn < real_block_n; sn++) {
			MTK_DEBUG_CALL_FUNC(printf("----------\n----- small n : %u\n----------\n", sn));
			MTK_DEBUG_PRINT_MATRIX(smem_A_ptr, m, real_block_n, DIM_MAX_M, "Input A");
			const auto gn = n_block * DIM_BLOCK_N + sn;

			// Copy y from A
			smem_y_ptr[threadIdx.x] = cutf::type::cast<T>(0.0f);
			if (threadIdx.x >= gn) {
				const auto index = DIM_MAX_M * sn + threadIdx.x;
				smem_y_ptr[threadIdx.x] = smem_A_ptr[index];
			}
			__syncthreads();
			MTK_DEBUG_PRINT_MATRIX(smem_y_ptr, 1, m, 1, "y (loaded)");

			// Compute norm2 of y and update y (y_i <- y_i +- norm(y);
			if (cutf::thread::get_warp_id() == gn / warp_size) {
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
			compute_reflection_0<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(smem_tmp_ptr, smem_y_ptr, smem_A_ptr);

			// Compute R
			compute_reflection_1<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(smem_A_ptr, smem_tmp_ptr, smem_y_ptr, t);

			// Compute W
			if (sn == 0) {
				smem_W_ptr[threadIdx.x] = smem_y_ptr[threadIdx.x] * t;
			} else {
				compute_w<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(smem_W_ptr + DIM_MAX_M * sn, smem_tmp_ptr, smem_y_ptr, smem_Y_ptr, smem_W_ptr, t);
			}
			smem_Y_ptr[sn * DIM_MAX_M + threadIdx.x] = smem_y_ptr[threadIdx.x];
		}
		// Store block A, W, Y, t to global memory
		copy_matrix_s2g<block_size, DIM_BLOCK_N, DIM_MAX_M>(gmem_a_ptr + lda * n_block * DIM_BLOCK_N, lda, smem_A_ptr, m, real_block_n);
		copy_matrix_s2g<block_size, DIM_BLOCK_N, DIM_MAX_M>(gmem_w_ptr + ldw * n_block * DIM_BLOCK_N, ldw, smem_W_ptr, m, real_block_n);
		copy_matrix_s2g<block_size, DIM_BLOCK_N, DIM_MAX_M>(gmem_y_ptr + ldy * n_block * DIM_BLOCK_N, ldy, smem_Y_ptr, m, real_block_n);

		// Update A
		for (std::size_t sub_n_block = n_block + 1; sub_n_block < num_n_blocks; sub_n_block++) {
			const unsigned real_block_n = umin(DIM_BLOCK_N, n - DIM_BLOCK_N * sub_n_block);
			copy_matrix_g2s<block_size, DIM_BLOCK_N, DIM_MAX_M>(smem_A_ptr, gmem_a_ptr + lda * sub_n_block * DIM_BLOCK_N, lda, m, real_block_n);
			update_a<compute_mode, DIM_MAX_M, DIM_BLOCK_N, DIM_MAX_M>(smem_A_ptr, smem_tmp_ptr, smem_W_ptr, smem_Y_ptr);
			copy_matrix_s2g<block_size, DIM_BLOCK_N, DIM_MAX_M>(gmem_a_ptr + lda * sub_n_block * DIM_BLOCK_N, lda, smem_A_ptr, m, real_block_n);
		}
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
__global__ void qr256x128_kernel(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n) {
	qr_kernel<compute_mode>(
			gmem_w_ptr, ldw,
			gmem_y_ptr, ldy,
			gmem_a_ptr, lda,
			m, n
			);
}
} // noname namespace

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::qr256x128(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n) {
	const unsigned block_size = 256;
	const unsigned smem_size = 58368; //[B]
	cudaFuncSetAttribute(qr256x128_kernel<compute_mode>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
	qr256x128_kernel<compute_mode><<<1, block_size, smem_size>>>(
			gmem_w_ptr, ldw,
			gmem_y_ptr, ldy,
			gmem_a_ptr, lda,
			m, n
			);
}

#define QR256X128_INSTANCE(compute_mode) \
template void mtk::tsqr_tc::qr256x128<compute_mode>( \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const std::size_t, \
		const std::size_t)

QR256X128_INSTANCE(mtk::tsqr_tc::compute_mode::fp32_hmma_cor);
