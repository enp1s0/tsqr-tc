#ifndef __TSQR_TC_DETAIL_TYPE_HPP__
#define __TSQR_TC_DETAIL_TYPE_HPP__
#include "constant.hpp"
namespace mtk {
namespace tsqr_tc {
namespace detail {
template <mtk::tsqr_tc::compute_mode::type compute_mode>
struct get_w_type {using type = float;};

template <mtk::tsqr_tc::compute_mode::type compute_mode>
struct get_y_type {using type = float;};

template <mtk::tsqr_tc::compute_mode::type compute_mode>
struct get_t_type {using type = float;};

template <mtk::tsqr_tc::compute_mode::type compute_mode>
struct get_a_type {using type = float;};

} // namespace detail
} // namespace tsqr_tc
} // namespace mtk
#endif
