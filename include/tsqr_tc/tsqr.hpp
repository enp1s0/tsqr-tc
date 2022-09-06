#ifndef __MTK_TSQR_TC_TSQR_HPP__
#define __MTK_TSQR_TC_TSQR_HPP__
#include "batchedqr.hpp"
#include <string>
#include <stdexcept>
namespace mtk {
namespace tsqr_tc {
enum wmem_alloc_location_t {
	host,
	device
};
template <mtk::tsqr_tc::compute_mode::type compute_mode>
class tsqr_buffer {
public:
	using buffer_type = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	static const std::size_t BQR_MAX_M = 256lu;
	static const std::size_t BQR_MAX_N = 128lu;
private:
	std::size_t m, n;
	buffer_type* r_buffer_ptr;
	buffer_type* w_buffer_ptr;
	buffer_type* y_buffer_ptr;
	std::size_t* index_buffer_ptr;
	std::size_t* index_buffer_host_ptr;
	wmem_alloc_location_t malloc_location;
public:
	tsqr_buffer(const std::size_t m, const std::size_t n) : m(m), n(n) {
		if (n > BQR_MAX_N) {
			std::runtime_error("[TSQR-TC] row size `n` must be equal or smaller than " + std::to_string(BQR_MAX_N));
		}
	}
	~tsqr_buffer();

	void allocate(const wmem_alloc_location_t loc = device);
	void free();

	void set_indices(cudaStream_t cuda_stream = 0);

	buffer_type* get_r_buffer_ptr() const {return r_buffer_ptr;}
	buffer_type* get_w_buffer_ptr() const {return w_buffer_ptr;}
	buffer_type* get_y_buffer_ptr() const {return y_buffer_ptr;}
	std::size_t* get_index_buffer_ptr() const {return index_buffer_ptr;}
	std::size_t* get_index_buffer_host_ptr() const {return index_buffer_host_ptr;}

	std::size_t get_r_buffer_count() const;
	std::size_t get_y_buffer_count() const;
	std::size_t get_w_buffer_count() const;
	std::size_t get_index_buffer_count() const;
	std::size_t get_split_count() const;

	std::size_t get_buffer_size() const;
};

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void tsqr(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const q_ptr, const std::size_t ld_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const r_ptr, const std::size_t ld_R,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const a_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		mtk::tsqr_tc::tsqr_buffer<compute_mode>& buffer,
		const cudaStream_t cuda_stream = 0
		);
} // namespace tsqr_tc
} // namespace mtk
#endif
