#ifndef __MTK_TSQR_TC_TEST_UTILS_HPP__
#define __MTK_TSQR_TC_TEST_UTILS_HPP__
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <tsqr_tc/detail/constant.hpp>
#include <string>

namespace mtk {
namespace tsqr_tc {
namespace test_utils {
template <mtk::tsqr_tc::compute_mode::type compute_mode>
inline const char* get_mode_name();
template <> inline const char* get_mode_name<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor   >() {return "fp32_fp16_hmma_cor"   ;}
template <> inline const char* get_mode_name<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor   >() {return "fp32_tf32_hmma_cor"   ;}
template <> inline const char* get_mode_name<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_no_cor>() {return "fp32_fp16_hmma_no_cor";}
template <> inline const char* get_mode_name<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_no_cor>() {return "fp32_tf32_hmma_no_cor";}
template <> inline const char* get_mode_name<mtk::tsqr_tc::compute_mode::fp32_no_tc           >() {return "fp32_no_tc"           ;}

template <class T>
double compute_residual_in_dp(
		const T* const dR_ptr, const std::size_t ld_R,
		const T* const dW_ptr, const std::size_t ld_W,
		const T* const dY_ptr, const std::size_t ld_Y,
		const T* const dA_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		);

template <class T>
double compute_residual_in_dp(
		const T* const dQ_ptr, const std::size_t ld_Q,
		const T* const dR_ptr, const std::size_t ld_R,
		const T* const dA_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		);

template <class T>
double compute_orthogonality_in_dp(
		const T* const dW_ptr, const std::size_t ld_W,
		const T* const dY_ptr, const std::size_t ld_Y,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		);

template <class T>
double compute_orthogonality_in_dp(
		const T* const dQ_ptr, const std::size_t ld_Q,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		);

template <class T>
void qr_cublas(
		T* const dQ_ptr, const std::size_t ld_Q,
		T* const dR_ptr, const std::size_t ld_R,
		const T* const dA_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		cusolverDnHandle_t const cusolver_handle
		);

template <class T>
void test_performance_cusolver(
		const std::size_t m,
		const std::size_t n,
		const unsigned test_count
		);

template <class T>
void test_performance_cublas_bqr(
		const std::size_t m, const std::size_t n,
		const unsigned batch_size_from,
		const unsigned batch_size_to,
		const unsigned test_count
		);


inline std::size_t compute_complexity(const std::size_t m, const std::size_t n) {
	return 4 * m * n * n + 80 * m * n;
};
} // namespace test_utils
} // namespace tsqr_tc
} // namespace mtk
#endif
