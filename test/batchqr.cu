#include <iostream>
#include <random>
#include <chrono>
#include <cutf/memory.hpp>
#include <tsqr_tc/batchedqr.hpp>
#include "utils.hpp"

constexpr unsigned batch_size_from = 1lu << 5;
constexpr unsigned batch_size_to   = 1lu << 14;
constexpr unsigned test_m = 256;
constexpr unsigned test_n = 128;
constexpr unsigned test_count = 16;

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_performance(const unsigned m, const unsigned n, const unsigned batch_size_from, const unsigned batch_size_to, const unsigned test_count) {
	using T = float;

	auto da_uptr = cutf::memory::get_device_unique_ptr<T>(m * n * batch_size_to);
	auto ha_uptr = cutf::memory::get_host_unique_ptr<T>  (m * n * batch_size_to);
	auto dw_uptr = cutf::memory::get_device_unique_ptr<T>(m * n * batch_size_to);
	auto dy_uptr = cutf::memory::get_device_unique_ptr<T>(m * n * batch_size_to);
	auto dr_uptr = cutf::memory::get_device_unique_ptr<T>(n * n * batch_size_to);

	auto dia_uptr = cutf::memory::get_device_unique_ptr<std::size_t>(batch_size_to + 1);
	auto hia_uptr = cutf::memory::get_host_unique_ptr  <std::size_t>(batch_size_to + 1);

	for (unsigned batch_size = batch_size_from; batch_size <= batch_size_to; batch_size <<= 1) {
		double average = 0.0;
		for (unsigned c = 0; c < test_count; c++) {
#pragma omp parallel
			{
				std::mt19937 mt(std::random_device{}());
				std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
#pragma omp for
				for (unsigned i = 0; i < m * n * batch_size; i++) {
					ha_uptr.get()[i] = dist(mt);
				}
			}
			cutf::memory::copy(da_uptr.get(), ha_uptr.get(), m * n * batch_size);
			for (unsigned i = 0; i < batch_size + 1; i++) {
				hia_uptr.get()[i] = i * m;
			}
			cutf::memory::copy(dia_uptr.get(), hia_uptr.get(), batch_size);

			auto start_clock = std::chrono::high_resolution_clock::now();

			mtk::tsqr_tc::qr256x128_batched<compute_mode>(
					dw_uptr.get(), m * batch_size,
					dy_uptr.get(), m * batch_size,
					dr_uptr.get(), n * batch_size,
					da_uptr.get(), m * batch_size,
					n, batch_size,
					dia_uptr.get()
					);
			CUTF_CHECK_ERROR(cudaDeviceSynchronize());

			auto end_clock = std::chrono::high_resolution_clock::now();

			const auto elasped_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
			average += elasped_time;
		}

		average /= test_count;

		std::printf("%u,%s,%e\n", batch_size, mtk::tsqr_tc::test_utils::get_mode_name<compute_mode>(), average);
	}
}

int main() {
	std::printf("batch_size,mode,time\n");
	test_performance<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor   >(test_m, test_n, batch_size_from, batch_size_to, test_count);
	test_performance<mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_no_cor>(test_m, test_n, batch_size_from, batch_size_to, test_count);
	test_performance<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_cor   >(test_m, test_n, batch_size_from, batch_size_to, test_count);
	test_performance<mtk::tsqr_tc::compute_mode::fp32_tf32_hmma_no_cor>(test_m, test_n, batch_size_from, batch_size_to, test_count);
	test_performance<mtk::tsqr_tc::compute_mode::fp32_no_tc           >(test_m, test_n, batch_size_from, batch_size_to, test_count);
	mtk::tsqr_tc::test_utils::test_performance_cublas_bqr<float       >(test_m, test_n, batch_size_from, batch_size_to, test_count);
}
