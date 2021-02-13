#ifndef __MTK_TSQR_TC_TSQR_HPP__
#define __MTK_TSQR_TC_TSQR_HPP__
#include "batchedqr.hpp"
namespace mtk {
namespace tsqr_tc {
template <mtk::tsqr_tc::compute_mode::type compute_mode>
class tsqr_buffer {
	using buffer_type = typename mtk::tsqr_tc::detail::get_type<compute_mode>;
	std::size_t m, n;
	buffer_type* r_buffer_ptr;
	buffer_type* w_buffer_ptr;
	buffer_type* y_buffer_ptr;
	std::size_t* index_buffer_ptr;
public:
	tsqr_buffer(const std::size_t m, const std::size_t n) : m(m), n(n) {}
	~tsqr_buffer();

	void allocate();
	void free();

	buffer_type* get_r_buffer_ptr() const {return r_buffer_ptr;}
	buffer_type* get_w_buffer_ptr() const {return w_buffer_ptr;}
	buffer_type* get_y_buffer_ptr() const {return y_buffer_ptr;}
	std::size_t* get_index_buffer_ptr() const {return index_buffer_ptr;}
};
} // namespace tsqr_tc
} // namespace mtk
#endif
