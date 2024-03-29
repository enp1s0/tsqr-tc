#include <iostream>
#include <random>
#include <chrono>
#include <tsqr_tc/tsqr.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include <cutf/stream.hpp>
#include <cutf/debug/matrix.hpp>

#include "utils.hpp"

constexpr float rand_abs_max = 1.0f;
constexpr unsigned test_count = 16;
constexpr std::size_t min_m_log = 10;
constexpr std::size_t max_m_log = 20;
constexpr std::size_t n = 128;

//#define MTK_DEBUG_PRINT

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_accuracy(const std::size_t m, const std::size_t n, const unsigned test_count) {
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
#ifdef MTK_DEBUG_PRINT
	auto hQ_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hR_uptr = cutf::memory::get_host_unique_ptr<compute_t>(n * n);
#endif

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dQ_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<compute_t>(n * n);

	auto cuda_stream_uptr = cutf::stream::get_stream_unique_ptr();

	mtk::tsqr_tc::tsqr_buffer<compute_mode> tsqr_buffer(m, n);
	tsqr_buffer.allocate();
	tsqr_buffer.set_indices(*cuda_stream_uptr.get());

	double orthogonality = 0.;
	double residual = 0.;

	for (unsigned c = 0; c < test_count; c++) {
		// initialize input matrix
		{
			std::mt19937 mt(std::random_device{}());
			std::uniform_real_distribution<float> dist(-rand_abs_max, rand_abs_max);
			for (unsigned i = 0; i < m * n; i++) {
				hA_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
			}
		}
#ifdef MTK_DEBUG_PRINT
		cutf::debug::print::print_numpy_matrix(hA_uptr.get(), m, n, "Input_A");
#endif

		cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

		mtk::tsqr_tc::tsqr(
				dQ_uptr.get(), m,
				dR_uptr.get(), n,
				dA_uptr.get(), m,
				m, n,
				tsqr_buffer,
				*cuda_stream_uptr.get()
				);
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());

		auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
		cublasSetStream(*cublas_handle.get(), *cuda_stream_uptr.get());

		const auto o = mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp(
				dQ_uptr.get(), m,
				m, n,
				*cublas_handle.get()
				);
		const auto r = mtk::tsqr_tc::test_utils::compute_residual_in_dp(
				dQ_uptr.get(), m,
				dR_uptr.get(), n,
				dA_uptr.get(), m,
				m, n,
				*cublas_handle.get()
				);
		residual += r;
		orthogonality += o;
#ifdef MTK_DEBUG_PRINT
		cutf::memory::copy(hR_uptr.get(), dR_uptr.get(), n * n);
		cutf::memory::copy(hQ_uptr.get(), dQ_uptr.get(), m * n);
		cutf::debug::print::print_numpy_matrix(hR_uptr.get(), n, n, "Output_R");
		cutf::debug::print::print_numpy_matrix(hQ_uptr.get(), m, n, "Outout_Q");
#endif
	}

	residual /= test_count;
	orthogonality /= test_count;

	std::printf("%lu,%lu,%s,%e,%e\n", m, n, mtk::tsqr_tc::test_utils::get_mode_name<compute_mode>(), residual, orthogonality);
	std::fflush(stdout);
}

template <class T>
void test_accuracy_cusolver(const std::size_t m, const std::size_t n, const unsigned test_count) {
	auto hA_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);
#ifdef MTK_DEBUG_PRINT
	auto hQ_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);
	auto hR_uptr = cutf::memory::get_host_unique_ptr<T>(n * n);
#endif

	auto dA_uptr = cutf::memory::get_device_unique_ptr<T>(m * n);
	auto dQ_uptr = cutf::memory::get_device_unique_ptr<T>(m * n);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<T>(n * n);

	double orthogonality = 0.;
	double residual = 0.;

	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();

	for (unsigned c = 0; c < test_count; c++) {
		// initialize input matrix
		{
			std::mt19937 mt(std::random_device{}());
			std::uniform_real_distribution<float> dist(-rand_abs_max, rand_abs_max);
			for (unsigned i = 0; i < m * n; i++) {
				hA_uptr.get()[i] = cutf::type::cast<T>(dist(mt));
			}
		}
#ifdef MTK_DEBUG_PRINT
		cutf::debug::print::print_numpy_matrix(hA_uptr.get(), m, n, "Input_A");
#endif

		cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

		mtk::tsqr_tc::test_utils::qr_cublas(
				dQ_uptr.get(), m,
				dR_uptr.get(), n,
				dA_uptr.get(), m,
				m, n,
				*cusolver_handle.get()
				);
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());

		auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

		const auto o = mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp(
				dQ_uptr.get(), m,
				m, n,
				*cublas_handle.get()
				);
		const auto r = mtk::tsqr_tc::test_utils::compute_residual_in_dp(
				dQ_uptr.get(), m,
				dR_uptr.get(), n,
				dA_uptr.get(), m,
				m, n,
				*cublas_handle.get()
				);
		residual += r;
		orthogonality += o;
#ifdef MTK_DEBUG_PRINT
		cutf::memory::copy(hR_uptr.get(), dR_uptr.get(), n * n);
		cutf::memory::copy(hQ_uptr.get(), dQ_uptr.get(), m * n);
		cutf::debug::print::print_numpy_matrix(hR_uptr.get(), n, n, "Output_R");
		cutf::debug::print::print_numpy_matrix(hQ_uptr.get(), m, n, "Outout_Q");
#endif
	}

	residual /= test_count;
	orthogonality /= test_count;

	std::printf("%lu,%lu,cusolver,%e,%e\n", m, n, residual, orthogonality);
	std::fflush(stdout);
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_performance(const std::size_t m, const std::size_t n, const unsigned test_count) {
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dQ_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<compute_t>(n * n);

	mtk::tsqr_tc::tsqr_buffer<compute_mode> tsqr_buffer(m, n);
	tsqr_buffer.allocate();

	// initialize input matrix
	{
		std::mt19937 mt(std::random_device{}());
		std::uniform_real_distribution<float> dist(-rand_abs_max, rand_abs_max);
		for (unsigned i = 0; i < m * n; i++) {
			hA_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
		}
	}
	cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

	auto cuda_stream_uptr = cutf::stream::get_stream_unique_ptr();
	tsqr_buffer.set_indices(*cuda_stream_uptr.get());
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto start_clock = std::chrono::high_resolution_clock::now();
	for (unsigned c = 0; c < test_count; c++) {
		mtk::tsqr_tc::tsqr(
				dQ_uptr.get(), m,
				dR_uptr.get(), n,
				dA_uptr.get(), m,
				m, n,
				tsqr_buffer,
				*cuda_stream_uptr.get()
				);
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::high_resolution_clock::now();

	const double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;

	std::size_t complexity = 0lu;
	const std::size_t qr_complexity = mtk::tsqr_tc::test_utils::compute_complexity(m / tsqr_buffer.get_split_count(), n);
	const std::size_t gemm_complexity = 6 * 128 * 128 * 128;
	for (unsigned bs = 1; bs <= tsqr_buffer.get_split_count(); bs <<= 1) {
		complexity += qr_complexity * bs;
		complexity += gemm_complexity * bs;
	}

	std::printf("%lu,%lu,%s,%e,%e,%lu\n", m, n, mtk::tsqr_tc::test_utils::get_mode_name<compute_mode>(), elapsed_time, complexity / elapsed_time, tsqr_buffer.get_buffer_size());
	std::fflush(stdout);
}

int main() {
	std::printf("m,n,mode,residual,orthogonality\n");
	std::fflush(stdout);
	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_accuracy<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_accuracy<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_accuracy<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_no_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_accuracy<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_no_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_accuracy<mtk::tsqr_tc::compute_mode::fp32_no_tc           >(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_accuracy_cusolver<float>(1lu << lm, n, test_count);
	}

	std::printf("m,n,mode,time,performance,buffer_size\n");
	std::fflush(stdout);
	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_performance<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_performance<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_performance<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_no_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_performance<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_no_cor>(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		test_performance<mtk::tsqr_tc::compute_mode::fp32_no_tc           >(1lu << lm, n, test_count);
	}

	for (std::size_t lm = min_m_log; lm <= max_m_log; lm++) {
		mtk::tsqr_tc::test_utils::test_performance_cusolver<float>(1lu << lm, n, test_count);
	}
}
