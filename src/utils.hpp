#ifndef __MTK_TSQR_TC_MATRIX_UTILS_HPP__
#define __MTK_TSQR_TC_MATRIX_UTILS_HPP__

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
	for (unsigned whole_vec_len = vec_len * block_size / warp_size; whole_vec_len > vec_len; whole_vec_len >>= 1) {
		for (unsigned offset = 0; offset < whole_vec_len / 2; offset += block_size) {
			const auto index = offset + threadIdx.x;
			if (index > whole_vec_len / 2) break;

			smem_vec_ptr[index] += smem_vec_ptr[index + whole_vec_len / 2];
		}
		__syncthreads();
	}
}

} // namespace utils
} // namespace tsqr_tc
} // namespace mtk
#endif
