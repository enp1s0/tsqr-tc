#ifndef __MTK_TSQR_TC_TEST_UTILS_HPP__
#define __MTK_TSQR_TC_TEST_UTILS_HPP__
#include <cutf/cublas.hpp>

namespace mtk {
namespace tsqr_tc {
namespace test_utils {
template <class T>
double compute_residual_in_dp(
		const T* const dR_ptr,
		const T* const dW_ptr,
		const T* const dY_ptr,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		);
} // namespace test_utils
} // namespace tsqr_tc
} // namespace mtk
#endif
