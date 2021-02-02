#include <cstdint>
#include <mma.h>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include <cutf/thread.hpp>
#include <wmma_extension.hpp>

#include <tsqr_tc/detail/constant.hpp>
#include <tsqr_tc/detail/type.hpp>

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
			const auto v = cutf::type::cast<GMEM_T>(0.0f);
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
		nvcuda::wmma::mma_sync(frag_ytA[k]  , frag_a[k]  , frag_yt[k], frag_ytA  );
		nvcuda::wmma::mma_sync(frag_d_ytA[k], frag_d_a[k], frag_yt[k], frag_d_ytA);
		nvcuda::wmma::mma_sync(frag_d_ytA[k], frag_a[k], frag_d_yt[k], frag_d_ytA);
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
} // noname namespace
