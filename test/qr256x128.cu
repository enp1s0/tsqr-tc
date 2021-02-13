#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/debug/matrix.hpp>
#include <cutf/cublas.hpp>

#include <tsqr_tc/batchedqr.hpp>
#include "utils.hpp"

constexpr float rand_abs_max = 1.0f;

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_accuracy(const unsigned m, const unsigned n, const std::size_t batch_size) {
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n * batch_size);
	auto hW_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n * batch_size);
	auto hY_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n * batch_size);
	auto hI_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * m * batch_size);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n * batch_size);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n * batch_size);
	auto dW_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n * batch_size);
	auto dY_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n * batch_size);

	// initialize input matrix
#pragma omp parallel
	{
		std::mt19937 mt(std::random_device{}());
		std::uniform_real_distribution<float> dist(-rand_abs_max, rand_abs_max);
#pragma omp for
		for (unsigned i = 0; i < m * n * batch_size; i++) {
			hA_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
		}
	}

	cutf::debug::print::print_matrix(hA_uptr.get(), m, n, "A (input)");
	cutf::memory::copy(dR_uptr.get(), hA_uptr.get(), m * n);
	cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

	auto d_start_m_list = cutf::memory::get_device_unique_ptr<std::size_t>(batch_size + 1);
	auto h_start_m_list = cutf::memory::get_host_unique_ptr<std::size_t>(batch_size + 1);
	for (std::size_t i = 0; i < (batch_size + 1); i++) {
		h_start_m_list.get()[i] = i * m;
	}
	cutf::memory::copy(d_start_m_list.get(), h_start_m_list.get(), batch_size + 1);

	mtk::tsqr_tc::qr256x128_batched<compute_mode>(
			dW_uptr.get(), m * batch_size,
			dY_uptr.get(), m * batch_size,
			dR_uptr.get(), m * batch_size,
			n,
			batch_size,
			d_start_m_list.get()
			);

	cutf::memory::copy(hA_uptr.get(), dR_uptr.get(), m * n * batch_size);
	cutf::memory::copy(hW_uptr.get(), dW_uptr.get(), m * n * batch_size);
	cutf::memory::copy(hY_uptr.get(), dY_uptr.get(), m * n * batch_size);

	cutf::debug::print::print_matrix(hA_uptr.get(), m, n, "R (output)");
	cutf::debug::print::print_matrix(hW_uptr.get(), m, n, "W (output)");
	cutf::debug::print::print_matrix(hY_uptr.get(), m, n, "Y (output)");

	// Compute using cusolver
	if (batch_size == 1) {
		auto cusolver_handle = cutf::cusolver::get_cusolver_dn_unique_ptr();
		auto hR_cusolver_uptr = cutf::memory::get_host_unique_ptr<compute_t>(n * n);
		auto hQ_cusolver_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
		CUTF_CHECK_ERROR(cudaMemset(hQ_cusolver_uptr.get(), 0, m * n));
		CUTF_CHECK_ERROR(cudaMemset(hR_cusolver_uptr.get(), 0, m * n));
		mtk::tsqr_tc::test_utils::qr_cublas(
				hQ_cusolver_uptr.get(), m,
				hR_cusolver_uptr.get(), n,
				dA_uptr.get(), m,
				m, n,
				*cusolver_handle.get()
				);
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());
		cutf::debug::print::print_matrix(hR_cusolver_uptr.get(), n, n, "R (cusolver output)");
		cutf::debug::print::print_matrix(hQ_cusolver_uptr.get(), m, n, "Q (cusolver output)");
	}

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	const auto residual = mtk::tsqr_tc::test_utils::compute_residual_in_dp(
			dR_uptr.get(), m,
			dW_uptr.get(), m,
			dY_uptr.get(), m,
			dA_uptr.get(), m,
			m, n,
			*cublas_handle.get()
			);

	const auto orthogonality = mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp(
			dW_uptr.get(), m,
			dY_uptr.get(), m,
			m, n,
			*cublas_handle.get()
			);

	std::printf("residual = %e\n", residual);
	std::printf("orthogonality = %e\n", orthogonality);
}

int main() {
	test_accuracy<mtk::tsqr_tc::compute_mode::fp32_hmma_cor>(32, 32, 1);
}
