#ifndef __MTK_BATCHED_QR_HPP__
#define __MTK_BATCHED_QR_HPP__
#include <cstdint>
#include "detail/type.hpp"

namespace mtk {
namespace tsqr_tc {
template <mtk::tsqr_tc::compute_mode::type compute_mode>
void qr256x128(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_w_ptr, const std::size_t ldw,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_y_ptr, const std::size_t ldy,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_t_ptr,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_a_ptr, const std::size_t lda,
		const std::size_t m,
		const std::size_t n);
} // namespace tsqr_tc
} // namespace mtk

#endif
