#include <iostream>
#include <random>
#include <tsqr_tc/tsqr.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include <cutf/stream.hpp>
#include <cutf/debug/matrix.hpp>

#include "utils.hpp"

constexpr float rand_abs_max = 1.0f;
constexpr unsigned test_count = 16;

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void test_accuracy(const std::size_t m, const std::size_t n, const unsigned test_count) {
	using compute_t = typename mtk::tsqr_tc::detail::get_type<compute_mode>::type;
	auto hA_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hQ_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
	auto hR_uptr = cutf::memory::get_host_unique_ptr<compute_t>(n * n);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dQ_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<compute_t>(n * n);

	mtk::tsqr_tc::tsqr_buffer<compute_mode> tsqr_buffer(m, n);
	tsqr_buffer.allocate();

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

		cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

		auto cuda_stream_uptr = cutf::stream::get_stream_unique_ptr();

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
	}

	residual /= test_count;
	orthogonality /= test_count;

	std::printf("%lu,%lu,%s,%e,%e\n", m, n, mtk::tsqr_tc::test_utils::get_mode_name<compute_mode>(), residual, orthogonality);
	std::fflush(stdout);
}

int main() {
	std::printf("m,n,mode,residual,orthogonality\n");
	for (std::size_t lm = 10; lm < 23; lm++) {
		test_accuracy<mtk::tsqr_tc::compute_mode::fp32_hmma_cor>(1lu << lm, 64, test_count);
	}
}
