#include <iostream>
#include <chrono>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/debug/matrix.hpp>
#include <cutf/cublas.hpp>

#include <tsqr_tc/batchedqr.hpp>
#include "utils.hpp"

//#define MTK_PRINT_MATRICES

constexpr float rand_abs_max = 1.0f;

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_accuracy(const unsigned m, const unsigned n, const std::size_t batch_size) {
	std::printf("# --- TEST --- %s / %s\n", __FILE__, __func__);
	std::printf("%20s : %u x %u\n", "input size", m, n);
	std::printf("%20s : %lu\n", "batch size", batch_size);
	std::printf("%20s : %s\n", "compute_mode", mtk::tsqr_tc::test_utils::get_mode_name<compute_mode>());
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n * batch_size);
	auto hW_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n * batch_size);
	auto hY_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n * batch_size);
	auto hR_uptr = cutf::memory::get_host_unique_ptr<compute_t>(n * n * batch_size);
	auto hI_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * m * batch_size);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n * batch_size);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<compute_t>(n * n * batch_size);
	auto dW_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n * batch_size);
	auto dY_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n * batch_size);

	// initialize input matrix
	{
		std::mt19937 mt(std::random_device{}());
		std::uniform_real_distribution<float> dist(-rand_abs_max, rand_abs_max);
		for (unsigned i = 0; i < m * n * batch_size; i++) {
			hA_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
		}
	}

	cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n * batch_size);

	auto d_start_m_list = cutf::memory::get_device_unique_ptr<std::size_t>(batch_size + 1);
	auto h_start_m_list = cutf::memory::get_host_unique_ptr<std::size_t>(batch_size + 1);
	for (std::size_t i = 0; i < (batch_size + 1); i++) {
		h_start_m_list.get()[i] = i * m;
	}
	cutf::memory::copy(d_start_m_list.get(), h_start_m_list.get(), batch_size + 1);

	const auto start_clock = std::chrono::high_resolution_clock::now();
	mtk::tsqr_tc::qr256x128_batched<compute_mode>(
			dW_uptr.get(), m * batch_size,
			dY_uptr.get(), m * batch_size,
			dR_uptr.get(), n * batch_size,
			dA_uptr.get(), m * batch_size,
			n,
			batch_size,
			d_start_m_list.get()
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::high_resolution_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
	std::printf("%20s : %e [s]\n", "time", elapsed_time);
	const auto compute_complexity = [](const unsigned m, const unsigned n) -> unsigned {
		return 4 * m * n * n + 64 * m * n;
	};
	std::printf("%20s : %e [TFlop/s]\n", "performance", compute_complexity(m, n) * batch_size / elapsed_time / 1e12);

	cutf::memory::copy(hR_uptr.get(), dR_uptr.get(), n * n * batch_size);
	cutf::memory::copy(hW_uptr.get(), dW_uptr.get(), m * n * batch_size);
	cutf::memory::copy(hY_uptr.get(), dY_uptr.get(), m * n * batch_size);

#ifdef MTK_PRINT_MATRICES
	// Compute using cusolver
	for (std::size_t s = 0; s < batch_size; s++) {
		cutf::debug::print::print_numpy_matrix(hR_uptr.get(), n, n, "R (output)");
		cutf::debug::print::print_numpy_matrix(hW_uptr.get(), m, n, "W (output)");
		cutf::debug::print::print_numpy_matrix(hY_uptr.get(), m, n, "Y (output)");

		auto cusolver_handle = cutf::cusolver::get_cusolver_dn_unique_ptr();
		auto hR_cusolver_uptr = cutf::memory::get_host_unique_ptr<compute_t>(n * n);
		auto hQ_cusolver_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
		CUTF_CHECK_ERROR(cudaMemset(hQ_cusolver_uptr.get(), 0, m * n * batch_size));
		CUTF_CHECK_ERROR(cudaMemset(hR_cusolver_uptr.get(), 0, n * n * batch_size));
		mtk::tsqr_tc::test_utils::qr_cublas(
				hQ_cusolver_uptr.get() + s * m, m,
				hR_cusolver_uptr.get() + s * n, n,
				dA_uptr.get() + s * m, m,
				m, n,
				*cusolver_handle.get()
				);
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());
		cutf::debug::print::print_numpy_matrix(hR_cusolver_uptr.get(), n, n, "R (cusolver output)");
		cutf::debug::print::print_numpy_matrix(hQ_cusolver_uptr.get(), m, n, "Q (cusolver output)");
	}
#endif

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	double residual = 0.;
	double orthogonality = 0.;
	for (std::size_t s = 0; s < batch_size; s++) {
		const auto r = mtk::tsqr_tc::test_utils::compute_residual_in_dp(
				dR_uptr.get() + s * n, n * batch_size,
				dW_uptr.get() + s * m, m * batch_size,
				dY_uptr.get() + s * m, m * batch_size,
				dA_uptr.get() + s * m, m * batch_size,
				m, n,
				*cublas_handle.get()
				);

		const auto o = mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp(
				dW_uptr.get() + s * m, m * batch_size,
				dY_uptr.get() + s * m, m * batch_size,
				m, n,
				*cublas_handle.get()
				);

		residual += r;
		orthogonality += o;
	}
	residual /= batch_size;
	orthogonality /= batch_size;
	std::printf("%20s : %e\n", "residual", residual);
	std::printf("%20s : %e\n", "orthogonality", orthogonality);
}

int main() {
	test_accuracy<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor>(256, 256, 1lu << 12);
	test_accuracy<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor>(256, 256, 1lu << 12);
}
