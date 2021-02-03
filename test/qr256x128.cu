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
void test_accuracy(const unsigned m, const unsigned n) {
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hW_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hY_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hI_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * m);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dW_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dY_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);

	// initialize input matrix
#pragma omp parallel
	{
		std::mt19937 mt(std::random_device{}());
		std::uniform_real_distribution<float> dist(-rand_abs_max, rand_abs_max);
#pragma omp for
		for (unsigned i = 0; i < m * n; i++) {
			hA_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
		}
	}

	cutf::debug::print::print_matrix(hA_uptr.get(), m, n, "A (input)");
	cutf::memory::copy(dR_uptr.get(), hA_uptr.get(), m * n);
	cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

	mtk::tsqr_tc::qr256x128<compute_mode>(
			dW_uptr.get(), m,
			dY_uptr.get(), m,
			dR_uptr.get(), m,
			m, n
			);

	cutf::memory::copy(hA_uptr.get(), dR_uptr.get(), m * n);
	cutf::memory::copy(hW_uptr.get(), dW_uptr.get(), m * n);
	cutf::memory::copy(hY_uptr.get(), dY_uptr.get(), m * n);

	cutf::debug::print::print_matrix(hA_uptr.get(), m, n, "R (output)");
	cutf::debug::print::print_matrix(hW_uptr.get(), m, n, "W (output)");
	cutf::debug::print::print_matrix(hY_uptr.get(), m, n, "Y (output)");

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	const auto residual = mtk::tsqr_tc::test_utils::compute_residual_in_dp(
			dR_uptr.get(), m,
			dW_uptr.get(), m,
			dY_uptr.get(), m,
			dA_uptr.get(), m,
			m, n,
			*cublas_handle.get()
			);
	std::printf("residual = %e\n", residual);

	const auto orthogonality = mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp(
			dW_uptr.get(), m,
			dY_uptr.get(), m,
			m, n,
			*cublas_handle.get()
			);
	std::printf("orthogonality = %e\n", orthogonality);
}

int main() {
	test_accuracy<mtk::tsqr_tc::compute_mode::fp32_hmma_cor>(256, 128);
}
