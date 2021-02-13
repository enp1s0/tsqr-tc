#ifndef __TSQR_TC_DETAIL_TYPE_HPP__
#define __TSQR_TC_DETAIL_TYPE_HPP__
#include "constant.hpp"
namespace mtk {
namespace tsqr_tc {
namespace detail {
template <mtk::tsqr_tc::compute_mode::type compute_mode>
struct get_type {using type = float;};
} // namespace detail
} // namespace tsqr_tc
} // namespace mtk
#endif
