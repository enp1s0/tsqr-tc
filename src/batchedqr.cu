#include <cstdint>
#include <cutf/type.hpp>
#include <cutf/math.hpp>
#include <cutf/thread.hpp>

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
} // noname namespace
