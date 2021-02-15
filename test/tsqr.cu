#include <iostream>
#include <random>
#include <chrono>
#include <tsqr_tc/tsqr.hpp>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>

constexpr float rand_abs_max = 1.0f;

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_accuracy(const unsigned m, const unsigned n) {
	std::printf("# --- TEST --- %s / %s\n", __FILE__, __func__);
	std::printf("%20s : %u x %u\n", "input size", m, n);
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hQ_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hR_uptr = cutf::memory::get_host_unique_ptr<compute_t>(n * n);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dQ_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<compute_t>(n * n);

	// initialize input matrix
	{
		std::mt19937 mt(std::random_device{}());
		std::uniform_real_distribution<float> dist(-rand_abs_max, rand_abs_max);
		for (unsigned i = 0; i < m * n; i++) {
			hA_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
		}
	}

#ifdef MTK_PRINT_MATRICES
	cutf::debug::print::print_numpy_matrix(hA_uptr.get(), m, n, "A");
#endif
	cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

	mtk::tsqr_tc::tsqr_buffer<compute_mode> tsqr_buffer(m, n);
	tsqr_buffer.allocate();
	std::printf("%20s : %e [MiB]\n", "buffer size", tsqr_buffer.get_buffer_size() / (1024. * 1024.));

	const auto start_clock = std::chrono::high_resolution_clock::now();
	mtk::tsqr_tc::tsqr(
			dQ_uptr.get(), m,
			dR_uptr.get(), n,
			dA_uptr.get(), m,
			m, n,
			tsqr_buffer
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::high_resolution_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
	std::printf("%20s : %e [s]\n", "time", elapsed_time);
}

int main() {
	test_accuracy<mtk::tsqr_tc::compute_mode::fp32_hmma_cor>(1lu << 10, 128);
}
