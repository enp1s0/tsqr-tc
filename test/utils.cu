#include <iostream>
#include <chrono>
#include <random>
#include "utils.hpp"
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/cusolver.hpp>

namespace {
template <class DST_T, class SRC_T>
__global__ void convert_matrix_kernel(
		DST_T* const dst_matrix_ptr, const std::size_t ld_dst,
		const SRC_T* const src_matrix_ptr, const std::size_t ld_src,
		const std::size_t m, const std::size_t n
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= m * n) return;
	const auto im = tid % m;
	const auto in = tid / m;

	dst_matrix_ptr[im + in * ld_dst] = cutf::type::cast<DST_T>(src_matrix_ptr[im + in * ld_src]);
}
template <class DST_T, class SRC_T>
void convert_matrix(
		DST_T* const dst_matrix_ptr, const std::size_t ld_dst,
		const SRC_T* const src_matrix_ptr, const std::size_t ld_src,
		const std::size_t m, const std::size_t n
		) {
	constexpr std::size_t block_size = 256;
	const auto num_threads = m * n;
	convert_matrix_kernel<DST_T, SRC_T><<<(num_threads + block_size - 1) / block_size, block_size>>>(
			dst_matrix_ptr, ld_dst,
			src_matrix_ptr, ld_src,
			m, n
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
}
}

template <class T>
double mtk::tsqr_tc::test_utils::compute_residual_in_dp(
		const T* const dR_ptr, const std::size_t ld_R,
		const T* const dW_ptr, const std::size_t ld_W,
		const T* const dY_ptr, const std::size_t ld_Y,
		const T* const dA_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		) {
	auto hR_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	auto hW_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	auto hY_dp_uptr = cutf::memory::get_host_unique_ptr<double>(n * n);
	auto hA_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	auto hTMP_dp_uptr = cutf::memory::get_host_unique_ptr<double>(n * n);
	for (unsigned i = 0; i < m * n; i++) hR_dp_uptr.get()[i] = 0.0f;
	convert_matrix(hR_dp_uptr.get(), m, dR_ptr, ld_R, n, n);
	convert_matrix(hW_dp_uptr.get(), m, dW_ptr, ld_W, m, n);
	convert_matrix(hY_dp_uptr.get(), n, dY_ptr, ld_Y, n, n);
	convert_matrix(hA_dp_uptr.get(), m, dA_ptr, ld_A, m, n);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto one = 1.0;
	const auto m_one = -1.0;
	const auto zero = 0.0;
	CUTF_CHECK_ERROR(
			cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_T, CUBLAS_OP_N,
				n, n, n,
				&one,
				hY_dp_uptr.get(), n,
				hR_dp_uptr.get(), m,
				&zero,
				hTMP_dp_uptr.get(), n
				)
			);
	CUTF_CHECK_ERROR(
			cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, n,
				&m_one,
				hW_dp_uptr.get(), m,
				hTMP_dp_uptr.get(), n,
				&one,
				hR_dp_uptr.get(), m
				)
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	// compute_diff
	double base_norm = 0.0;
	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: base_norm) reduction(+: diff_norm)
	for (std::size_t i = 0; i < m * n; i++) {
		const auto diff = hR_dp_uptr.get()[i] - hA_dp_uptr.get()[i];
		const auto base = hA_dp_uptr.get()[i];

		base_norm += base * base;
		diff_norm += diff * diff;
	}

	return std::sqrt(diff_norm / base_norm);
}

template
double mtk::tsqr_tc::test_utils::compute_residual_in_dp<float>(
		const float* const, const std::size_t,
		const float* const, const std::size_t,
		const float* const, const std::size_t,
		const float* const, const std::size_t,
		const std::size_t, const std::size_t,
		cublasHandle_t const
		);


template <class T>
double mtk::tsqr_tc::test_utils::compute_residual_in_dp(
		const T* const dQ_ptr, const std::size_t ld_Q,
		const T* const dR_ptr, const std::size_t ld_R,
		const T* const dA_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		) {
	auto hR_dp_uptr = cutf::memory::get_host_unique_ptr<double>(n * n);
	auto hQ_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	auto hA_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	convert_matrix(hR_dp_uptr.get(), n, dR_ptr, ld_R, n, n);
	convert_matrix(hQ_dp_uptr.get(), m, dQ_ptr, ld_Q, m, n);
	convert_matrix(hA_dp_uptr.get(), m, dA_ptr, ld_A, m, n);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	double base_norm = 0.0;
#pragma omp parallel for reduction(+: base_norm)
	for (std::size_t i = 0; i < m * n; i++) {
		const auto base = hA_dp_uptr.get()[i];

		base_norm += base * base;
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto one = 1.0;
	const auto m_one = -1.0;
	CUTF_CHECK_ERROR(
			cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, n,
				&one,
				hQ_dp_uptr.get(), m,
				hR_dp_uptr.get(), n,
				&m_one,
				hA_dp_uptr.get(), m
				)
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	// compute_diff
	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: diff_norm)
	for (std::size_t i = 0; i < m * n; i++) {
		const auto diff = hA_dp_uptr.get()[i];

		diff_norm += diff * diff;
	}

	return std::sqrt(diff_norm / base_norm);
}

template
double mtk::tsqr_tc::test_utils::compute_residual_in_dp<float>(
		const float* const, const std::size_t,
		const float* const, const std::size_t,
		const float* const, const std::size_t,
		const std::size_t, const std::size_t,
		cublasHandle_t const
		);


template <class T>
double mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp(
		const T* const dW_ptr, const std::size_t ld_W,
		const T* const dY_ptr, const std::size_t ld_Y,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		) {
	auto hW_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	auto hY_dp_uptr = cutf::memory::get_host_unique_ptr<double>(n * n);
	auto hQ_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	auto hE_dp_uptr = cutf::memory::get_host_unique_ptr<double>(n * n);
	convert_matrix(hW_dp_uptr.get(), m, dW_ptr, ld_W, m, n);
	convert_matrix(hY_dp_uptr.get(), n, dY_ptr, ld_Y, n, n);

	// initialize Q
#pragma omp parallel for
	for (std::size_t i = 0; i < m * n; i++) {
		hQ_dp_uptr.get()[i] = 0.0;
	}
	for (std::size_t i = 0; i < n; i++) {
		hQ_dp_uptr.get()[i * (1 + m)] = 1.0;
	}
	// initialize E
#pragma omp parallel for
	for (std::size_t i = 0; i < n * n; i++) {
		hE_dp_uptr.get()[i] = 0.0;
	}
	for (std::size_t i = 0; i < n; i++) {
		hE_dp_uptr.get()[i * (1 + n)] = 1.0;
	}

	const auto one = 1.0;
	const auto m_one = -1.0;
	CUTF_CHECK_ERROR(
			cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				m, n, n,
				&m_one,
				hW_dp_uptr.get(), m,
				hY_dp_uptr.get(), n,
				&one,
				hQ_dp_uptr.get(), m
				)
			);
	CUTF_CHECK_ERROR(
			cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_T, CUBLAS_OP_N,
				n, n, m,
				&m_one,
				hQ_dp_uptr.get(), m,
				hQ_dp_uptr.get(), m,
				&one,
				hE_dp_uptr.get(), n
				)
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: diff_norm)
	for (std::size_t i = 0; i < n * n; i++) {
		const auto diff = hE_dp_uptr.get()[i];
		diff_norm += diff * diff;
	}

	return std::sqrt(diff_norm / n);
}

template
double mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp<float>(
		const float* const, const std::size_t,
		const float* const, const std::size_t,
		const std::size_t, const std::size_t,
		cublasHandle_t const
		);


template <class T>
double mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp(
		const T* const dQ_ptr, const std::size_t ld_Q,
		const std::size_t m, const std::size_t n,
		cublasHandle_t const cublas_handle
		) {
	auto hQ_dp_uptr = cutf::memory::get_host_unique_ptr<double>(m * n);
	auto hE_dp_uptr = cutf::memory::get_host_unique_ptr<double>(n * n);
	convert_matrix(hQ_dp_uptr.get(), m, dQ_ptr, ld_Q, m, n);

	// initialize E
#pragma omp parallel for
	for (std::size_t i = 0; i < n * n; i++) {
		hE_dp_uptr.get()[i] = 0.0;
	}
	for (std::size_t i = 0; i < n; i++) {
		hE_dp_uptr.get()[i * (1 + n)] = 1.0;
	}

	const auto one = 1.0;
	const auto m_one = -1.0;
	CUTF_CHECK_ERROR(
			cutf::cublas::gemm(
				cublas_handle,
				CUBLAS_OP_T, CUBLAS_OP_N,
				n, n, m,
				&m_one,
				hQ_dp_uptr.get(), m,
				hQ_dp_uptr.get(), m,
				&one,
				hE_dp_uptr.get(), n
				)
			);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: diff_norm)
	for (std::size_t i = 0; i < n * n; i++) {
		const auto diff = hE_dp_uptr.get()[i];
		diff_norm += diff * diff;
	}

	return std::sqrt(diff_norm / n);
}

template
double mtk::tsqr_tc::test_utils::compute_orthogonality_in_dp<float>(
		const float* const, const std::size_t,
		const std::size_t, const std::size_t,
		cublasHandle_t const
		);

// CUSOLVER QR
namespace {
template <class T>
__global__ void cut_r_kernel(
		T* const dst, const std::size_t ld_dst,
		const T* const src, const std::size_t ld_src,
		const std::size_t n) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	const auto x = tid / n;
	const auto y = tid % n;

	if(y > x) return;

	dst[ld_dst * x + y] = src[ld_src * x + y];
}

template <class T>
void cut_r(T* const dst, const std::size_t ld_dst,
		const T* const src, const std::size_t ld_src,
		const std::size_t n) {
	constexpr std::size_t block_size = 256;
	cut_r_kernel<T><<<(n * n + block_size - 1) / block_size, block_size>>>(
			dst, ld_dst,
			src, ld_src,
			n
			);
}
}

template <class T>
void mtk::tsqr_tc::test_utils::qr_cublas(
		T* const dQ_ptr, const std::size_t ld_Q,
		T* const dR_ptr, const std::size_t ld_R,
		const T* const dA_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		cusolverDnHandle_t const cusolver_handle
		) {
	convert_matrix<T, T>(
			dQ_ptr, ld_Q,
			dA_ptr, ld_A,
			m, n
			);
	auto d_tau = cutf::memory::get_device_unique_ptr<T>(n * n);
	int geqrf_working_memory_size, gqr_working_memory_size;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
				cusolver_handle, m, n,
				dQ_ptr, ld_Q, &geqrf_working_memory_size
				));
	CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr_buffer_size(
				cusolver_handle, m, n, n,
				dQ_ptr, ld_Q, d_tau.get(), &gqr_working_memory_size
				));

	auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<T>(geqrf_working_memory_size);
	auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<T>(gqr_working_memory_size);
	auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
				cusolver_handle, m, n,
				dQ_ptr, ld_Q, d_tau.get(), d_geqrf_working_memory.get(),
				geqrf_working_memory_size, d_info.get()
				));
	cut_r(
			dR_ptr, ld_R,
			dQ_ptr, ld_A,
			n
			);

	CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr(
				cusolver_handle, m, n, n,
				dQ_ptr, ld_Q,
				d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
				d_info.get()
				));

}

template
void mtk::tsqr_tc::test_utils::qr_cublas<float>(
		float* const dQ_ptr, const std::size_t ld_Q,
		float* const dR_ptr, const std::size_t ld_R,
		const float* const dA_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		cusolverDnHandle_t const cusolver_handle
		);


template <class T>
void mtk::tsqr_tc::test_utils::test_performance_cusolver(const std::size_t m, const std::size_t n, const unsigned test_count) {
	auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
	auto hA_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);

	auto dA_uptr = cutf::memory::get_device_unique_ptr<T>(m * n);
	auto dQ_uptr = cutf::memory::get_device_unique_ptr<T>(m * n);
	auto dR_uptr = cutf::memory::get_device_unique_ptr<T>(n * n);

	auto d_tau = cutf::memory::get_device_unique_ptr<T>(n * n);
	int geqrf_working_memory_size, gqr_working_memory_size;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
				*cusolver_handle.get(), m, n,
				dQ_uptr.get(), m, &geqrf_working_memory_size
				));
	CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr_buffer_size(
				*cusolver_handle.get(), m, n, n,
				dQ_uptr.get(), m, d_tau.get(), &gqr_working_memory_size
				));

	auto d_geqrf_working_memory = cutf::memory::get_device_unique_ptr<T>(geqrf_working_memory_size);
	auto d_gqr_working_memory = cutf::memory::get_device_unique_ptr<T>(gqr_working_memory_size);
	auto d_info = cutf::memory::get_device_unique_ptr<int>(1);

	// initialize input matrix
	{
		std::mt19937 mt(std::random_device{}());
		std::uniform_real_distribution<T> dist(-1., 1.);
		for (unsigned i = 0; i < m * n; i++) {
			hA_uptr.get()[i] = cutf::type::cast<T>(dist(mt));
		}
	}
	cutf::memory::copy(dA_uptr.get(), hA_uptr.get(), m * n);

	const auto start_clock = std::chrono::high_resolution_clock::now();
	for (unsigned c = 0; c < test_count; c++) {

		CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
					*cusolver_handle.get(), m, n,
					dQ_uptr.get(), m, d_tau.get(), d_geqrf_working_memory.get(),
					geqrf_working_memory_size, d_info.get()
					));
		cut_r(
				dR_uptr.get(), n,
				dQ_uptr.get(), m,
				n
			 );

		CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr(
					*cusolver_handle.get(), m, n, n,
					dQ_uptr.get(), m,
					d_tau.get(), d_gqr_working_memory.get(), gqr_working_memory_size,
					d_info.get()
					));
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::high_resolution_clock::now();

	const double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6 / test_count;

	std::printf("%lu,%lu,cusolver,%e,0,%lu\n", m, n, elapsed_time, std::max(geqrf_working_memory_size, gqr_working_memory_size) * sizeof(T));
	std::fflush(stdout);
}

template
void mtk::tsqr_tc::test_utils::test_performance_cusolver<float >(const std::size_t, const std::size_t, const unsigned);


template <class T>
void mtk::tsqr_tc::test_utils::test_performance_cublas_bqr(
		const std::size_t m, const std::size_t n,
		const unsigned batch_size_from,
		const unsigned batch_size_to,
		const unsigned test_count
		) {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<T> dist(-1., 1.);

	auto da_uptr = cutf::memory::get_device_unique_ptr<T>(m * n * batch_size_to);
	auto ha_uptr = cutf::memory::get_host_unique_ptr<T>  (m * n * batch_size_to);
	auto dt_uptr = cutf::memory::get_device_unique_ptr<T>(    n * batch_size_to);
	auto ht_uptr = cutf::memory::get_host_unique_ptr<T>  (    n * batch_size_to);

	auto dia_uptr = cutf::memory::get_device_unique_ptr<T*>(batch_size_to);
	auto hia_uptr = cutf::memory::get_host_unique_ptr<T*>  (batch_size_to);
	auto dit_uptr = cutf::memory::get_device_unique_ptr<T*>(batch_size_to);
	auto hit_uptr = cutf::memory::get_host_unique_ptr<T*>  (batch_size_to);

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	for (unsigned batch_size = batch_size_from; batch_size <= batch_size_to; batch_size <<= 1) {
		double average = 0.0;
		for (unsigned c = 0; c < test_count; c++) {
			for (unsigned i = 0; i < m * n * batch_size; i++) {
				ha_uptr.get()[i] = dist(mt);
			}
			cutf::memory::copy(da_uptr.get(), ha_uptr.get(), m * n * batch_size);
			for (unsigned i = 0; i < batch_size; i++) {
				hia_uptr.get()[i] = da_uptr.get() + i * m;
				hit_uptr.get()[i] = dt_uptr.get() + i * n;
			}
			cutf::memory::copy(dia_uptr.get(), hia_uptr.get(), batch_size);
			cutf::memory::copy(dit_uptr.get(), hit_uptr.get(), batch_size);

			auto start_clock = std::chrono::high_resolution_clock::now();

			int info;
			CUTF_CHECK_ERROR(cutf::cublas::geqrf_batched(
						*cublas_handle.get(),
						m, n,
						dia_uptr.get(), m * batch_size,
						dit_uptr.get(),
						&info,
						batch_size
						));
			CUTF_CHECK_ERROR(cudaDeviceSynchronize());

			auto end_clock = std::chrono::high_resolution_clock::now();

			const auto elasped_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
			average += elasped_time;
		}

		average /= test_count;

		std::printf("%u,cusolver,%e,0\n", batch_size, average);
	}
}

template void mtk::tsqr_tc::test_utils::test_performance_cublas_bqr<float >(const std::size_t, const std::size_t, const unsigned, const unsigned, const unsigned);
