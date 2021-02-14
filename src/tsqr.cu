#include <cassert>
#include <tsqr_tc/tsqr.hpp>
#include <cutf/memory.hpp>

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::tsqr(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>* r_ptr, const std::size_t ld_R,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>* w_ptr, const std::size_t ld_W,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>* y_ptr, const std::size_t ld_Y,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>* a_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		mtk::tsqr_tc::tsqr_buffer<compute_mode>& buffer,
		const cudaStream_t cuda_stream
		) {
	assert(n <= 128);

	for (std::size_t i = 0; i < buffer.get_split_count() + 1; i++) {
		buffer.get_index_buffer_count()[i] = i * m / buffer.get_split_count();
	}
	cutf::memory::copy_async(buffer.get_index_buffer_ptr(), buffer.get_index_buffer_host_ptr(), buffer.get_index_buffer_count(), cuda_stream);
	for (std::size_t i = 0; i < buffer.get_split_count(); i++) {
		buffer.get_index_buffer_count()[i] = i * (2 * n);
	}

	unsigned r_ptr_index = 0;
	typename mtk::tsqr_tc::tsqr_buffer<compute_mode>::buffer_type r_buffer_list[2] = {
		buffer.get_r_buffer_ptr(),
		buffer.get_r_buffer_ptr() + buffer.get_split_count() * n * n
	};

	std::size_t wy_ptr_offset = 0;

	// Forwad computation
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
	wy_ptr_offset += m * n;
	r_ptr_index = 1 - r_ptr_index;

	cutf::memory::copy_async(buffer.get_index_buffer_ptr(), buffer.get_index_buffer_host_ptr(), buffer.get_index_buffer_count(), cuda_stream);

	for (std::size_t s = buffer.get_split_count(); s > 1; s >>= 1) {
		mtk::tsqr_tc::qr256x128_batched<compute_mode>(
				buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				r_buffer_list[r_ptr_index], n * s,
				r_buffer_list[1 - r_ptr_index], 2 * n * s,
				n,
				s,
				buffer.get_index_buffer_ptr(),
				cuda_stream
				);
		wy_ptr_offset += 2 * n * n * s;
		r_ptr_index = 1 - r_ptr_index;
	}
	mtk::tsqr_tc::qr256x128<compute_mode>(
			buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n,
			r_ptr, ld_R,
			2 * n, n,
			cuda_stream
			);
}
