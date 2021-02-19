#include <cmath>
#include <algorithm>
#include <tsqr_tc/tsqr.hpp>
#include <cutf/memory.hpp>

template <mtk::tsqr_tc::compute_mode::type compute_mode>
mtk::tsqr_tc::tsqr_buffer<compute_mode>::~tsqr_buffer() {
	this->free();
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
std::size_t mtk::tsqr_tc::tsqr_buffer<compute_mode>::get_split_count() const {
	const auto log2_m = std::max(std::log2(static_cast<double>(m)), std::log2(static_cast<double>(BQR_MAX_M))) - std::log2(static_cast<double>(BQR_MAX_M));
	const unsigned log2_split_count = std::ceil(log2_m);
	return 1lu << log2_split_count;
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
std::size_t mtk::tsqr_tc::tsqr_buffer<compute_mode>::get_r_buffer_count() const {
	const auto split_count = get_split_count();
	return (split_count + (split_count + 1) / 2) * n * n;
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
std::size_t mtk::tsqr_tc::tsqr_buffer<compute_mode>::get_w_buffer_count() const {
	const auto split_count = get_split_count();
	return m * n + (2 * split_count - 1) * ((2 * n) * n);
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
std::size_t mtk::tsqr_tc::tsqr_buffer<compute_mode>::get_y_buffer_count() const {
	return get_w_buffer_count();
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
std::size_t mtk::tsqr_tc::tsqr_buffer<compute_mode>::get_index_buffer_count() const {
	return get_split_count() + 1;
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
std::size_t mtk::tsqr_tc::tsqr_buffer<compute_mode>::get_buffer_size() const {
	return sizeof(typename mtk::tsqr_tc::tsqr_buffer<compute_mode>::buffer_type) * (get_w_buffer_count() + get_y_buffer_count() + get_r_buffer_count())
		+ sizeof(std::size_t) * get_index_buffer_count();
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::tsqr_buffer<compute_mode>::set_indices(cudaStream_t cuda_stream) {
	for (std::size_t i = 0; i < this->get_split_count() + 1; i++) {
		this->get_index_buffer_host_ptr()[i] = i * m / this->get_split_count();
	}
	cutf::memory::copy_async(this->get_index_buffer_ptr(), this->get_index_buffer_host_ptr(), this->get_index_buffer_count(), cuda_stream);
}


template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::tsqr_buffer<compute_mode>::allocate() {
	CUTF_CHECK_ERROR(cudaMalloc(&r_buffer_ptr, sizeof(typename mtk::tsqr_tc::tsqr_buffer<compute_mode>::buffer_type) * get_r_buffer_count()));
	CUTF_CHECK_ERROR(cudaMalloc(&w_buffer_ptr, sizeof(typename mtk::tsqr_tc::tsqr_buffer<compute_mode>::buffer_type) * get_w_buffer_count()));
	CUTF_CHECK_ERROR(cudaMalloc(&y_buffer_ptr, sizeof(typename mtk::tsqr_tc::tsqr_buffer<compute_mode>::buffer_type) * get_y_buffer_count()));
	CUTF_CHECK_ERROR(cudaMalloc(&index_buffer_ptr, sizeof(std::size_t) * get_index_buffer_count()));
	CUTF_CHECK_ERROR(cudaMallocHost(&index_buffer_host_ptr, sizeof(std::size_t) * get_index_buffer_count()));
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::tsqr_buffer<compute_mode>::free() {
	auto free_func = [](auto* ptr) {
		if (ptr != nullptr) {
			CUTF_CHECK_ERROR(cudaFree(ptr));
			ptr = nullptr;
		}
	};
	auto free_host_func = [](auto* ptr) {
		if (ptr != nullptr) {
			CUTF_CHECK_ERROR(cudaFreeHost(ptr));
			ptr = nullptr;
		}
	};
	free_func(r_buffer_ptr);
	free_func(w_buffer_ptr);
	free_func(y_buffer_ptr);
	free_func(index_buffer_ptr);
	free_host_func(index_buffer_host_ptr);
}

template class mtk::tsqr_tc::tsqr_buffer<mtk::tsqr_tc::compute_mode::fp32_hmma_cor>;
