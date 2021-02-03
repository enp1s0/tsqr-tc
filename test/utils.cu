#include "utils.hpp"
#include <cutf/memory.hpp>
#include <cutf/type.hpp>

namespace {
template <class DST_T, class SRC_T>
__global__ void convert_matrix_kernel(
		DST_T* const dst_matrix_ptr, const std::size_t ld_dst,
		const SRC_T* const src_matrix_ptr, const std::size_t ld_src,
		const std::size_t m, const std::size_t n
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= m * n) return;
	const auto im = tid % m;
	const auto in = tid / m;

	dst_matrix_ptr[im + in * ld_dst] = cutf::type::cast<DST_T>(src_matrix_ptr[im + in * ld_src]);
}
template <class DST_T, class SRC_T>
void convert_matrix(
		DST_T* const dst_matrix_ptr, const std::size_t ld_dst,
		const SRC_T* const src_matrix_ptr, const std::size_t ld_src,
		const std::size_t m, const std::size_t n
		) {
	constexpr std::size_t block_size = 256;
	const auto num_threads = m * n;
	convert_matrix<DST_T, SRC_T><<<(num_threads + block_size - 1) / block_size, block_size>>>(
			dst_matrix_ptr, ld_dst,
			src_matrix_ptr, ld_src,
			m, n
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
}
}
