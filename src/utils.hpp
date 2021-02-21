#ifndef __MTK_TSQR_TC_MATRIX_UTILS_HPP__
#define __MTK_TSQR_TC_MATRIX_UTILS_HPP__
#include <cutf/type.hpp>

namespace mtk {
namespace tsqr_tc {
namespace utils {

// This function accumulates vectors on shared memory.
// Restrictions:
// count == block_size / war_size
// output_ptr == inpute_ptr
template <unsigned block_size, class T>
__device__ inline void accumulate_vectors(T* const smem_vec_ptr, const unsigned vec_len) {
	constexpr unsigned warp_size = 32;
	if (vec_len <= block_size) {
		if (threadIdx.x < vec_len) {
			auto v = cutf::type::cast<T>(0.0f);
			for (unsigned i = 0; i < block_size / warp_size; i++) {
				v += smem_vec_ptr[i * vec_len + threadIdx.x];
			}
			smem_vec_ptr[threadIdx.x] = v;
		}
	} else {
		for (unsigned j = 0; j < vec_len; j += block_size) {
			auto v = cutf::type::cast<T>(0.0f);
			for (unsigned i = 0; i < block_size / warp_size; i++) {
				v += smem_vec_ptr[j + i * vec_len + threadIdx.x];
			}
			smem_vec_ptr[j + threadIdx.x] = v;
		}
	}
	__syncthreads();
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


} // namespace utils
} // namespace tsqr_tc
} // namespace mtk
#endif
