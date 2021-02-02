#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>

#include <tsqr_tc/batchedqr.hpp>

constexpr float rand_abs_max = 1.0f;

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_accuracy(const unsigned m, const unsigned n) {
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hW_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hY_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto ht_uptr = cutf::memory::get_host_unique_ptr<compute_t>(n);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dW_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dY_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dt_uptr = cutf::memory::get_device_unique_ptr<compute_t>(n);

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

	cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

	mtk::tsqr_tc::qr256x128<compute_mode>(
			dW_uptr.get(), m,
			dY_uptr.get(), m,
			dt_uptr.get(),
			dA_uptr.get(), m,
			m, n
			);
}

int main() {
	test_accuracy<mtk::tsqr_tc::compute_mode::fp32_hmma_cor>(256, 128);
}
