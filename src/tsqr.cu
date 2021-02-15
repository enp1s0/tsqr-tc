#include <cassert>
#include <tsqr_tc/tsqr.hpp>
#include <cutf/memory.hpp>
#include <cutf/thread.hpp>
#include <cutf/type.hpp>
#include <wmma_extension.hpp>

namespace {
constexpr unsigned warp_size = 32;

template <unsigned block_size, unsigned DIM_N, int TRANSPOSE, class DST_T, class SRC_T>
__device__ void copy_B_g2s(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr, const std::size_t ld_src,
		const unsigned n
		) {
	constexpr auto num_warps = block_size / warp_size;
	for (unsigned bm = 0; bm < DIM_N; bm += warp_size) {
		const auto real_bm = min(warp_size, n - bm);
		if (real_bm == warp_size) {
			for (unsigned bn = 0; bn < n; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				auto v = cutf::type::cast<SRC_T>(0.0f);
				if (gn < n) {
					v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
				}
				if constexpr (TRANSPOSE == 0) {
					dst_ptr[gn + (cutf::thread::get_lane_id() + bm) * DIM_N] = cutf::type::cast<DST_T>(v);
				} else {
					dst_ptr[gn * DIM_N + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		} else {
			for (unsigned bn = 0; bn < DIM_N; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				auto v = cutf::type::cast<SRC_T>(0.0f);
				if (gn < n && cutf::thread::get_lane_id() < real_bm) {
					v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
				}
				if constexpr (TRANSPOSE == 0) {
					dst_ptr[gn + (cutf::thread::get_lane_id() + bm) * DIM_N] = cutf::type::cast<DST_T>(v);
				} else {
					dst_ptr[gn * DIM_N + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		}
	}
}

template <unsigned block_size, unsigned DIM_XP, unsigned DIM_N, class DST_T, class SRC_T>
__device__ void copy_matrix_g2s_XPxN(
		DST_T* const dst_ptr,
		const SRC_T* const src_ptr, const std::size_t ld_src,
		const unsigned m, const unsigned n
		) {
	constexpr auto num_warps = block_size / warp_size;
	for (unsigned bm = 0; bm < m; bm += warp_size) {
		const auto real_bm = min(warp_size, m - bm);
		if (real_bm == warp_size) {
			for (unsigned bn = 0; bn < n; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				auto v = cutf::type::cast<SRC_T>(0.0f);
				if (gn < n) {
					v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
				}
				dst_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
			}
		} else {
			for (unsigned bn = 0; bn < n; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				auto v = cutf::type::cast<SRC_T>(0.0f);
				if (gn < n && cutf::thread::get_lane_id() < real_bm) {
					v = src_ptr[gn * ld_src + cutf::thread::get_lane_id() + bm];
				}
				dst_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
			}
		}
	}
}

template <unsigned block_size, unsigned DIM_XP, unsigned DIM_N, class DST_T, class SRC_T>
__device__ void copy_matrix_s2g_XPxN(
		DST_T* const dst_ptr, const std::size_t ld_dst,
		const SRC_T* const src_ptr,
		const unsigned m, const unsigned n
		) {
	constexpr auto num_warps = block_size / warp_size;
	for (unsigned bm = 0; bm < m; bm += warp_size) {
		const auto real_bm = min(warp_size, m - bm);
		if (real_bm == warp_size) {
			for (unsigned bn = 0; bn < n; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				if (gn < n) {
					const auto v = src_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm];
					dst_ptr[gn * ld_dst + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		} else {
			for (unsigned bn = 0; bn < n; bn += num_warps) {
				const auto gn = bn + cutf::thread::get_warp_id();
				if (gn < n && cutf::thread::get_lane_id() < real_bm) {
					const auto v = src_ptr[gn * DIM_XP + cutf::thread::get_lane_id() + bm];
					dst_ptr[gn * ld_dst + cutf::thread::get_lane_id() + bm] = cutf::type::cast<DST_T>(v);
				}
			}
		}
	}
}

template <int A_MINUS, int B_TRANS, int C_EXIST>
__device__ void gemm_MxNxN_core_fp32_hmma_cor(
		float* const gmem_D_ptr, const std::size_t ld_D,
		const float* const gmem_A_ptr, const std::size_t ld_A,
		const float* const gmem_B_ptr, const std::size_t ld_B,
		const float* const gmem_C_ptr, const std::size_t ld_C,
		const std::size_t m, const std::size_t n
		) {
	constexpr unsigned block_size = 256;
	constexpr std::size_t DIM_N = 128;
	constexpr std::size_t DIM_BLOCK_M = 64;
	constexpr std::size_t K_BLOCKING = 64;
	constexpr std::size_t DIM_TC = 16;
	constexpr std::size_t NUM_BLOCKINGS = K_BLOCKING / DIM_TC;

	constexpr float cor_scale = 1024.f;

	extern __shared__ float smem[];
	float* const smem_B_ptr = smem;
	float* const smem_A_ptr = smem_B_ptr + DIM_N * DIM_N;

	// Load B
	copy_B_g2s<block_size, DIM_N, B_TRANS>(
			smem_B_ptr,
			gmem_B_ptr, ld_B,
			n
			);
	for (std::size_t bm = 0; bm < m; bm += DIM_BLOCK_M) {
		nvcuda::wmma::fragment<nvcuda::wmma::accumulator, DIM_TC, DIM_TC, DIM_TC, float> frag_C[DIM_BLOCK_M / DIM_TC], frag_d_C[DIM_BLOCK_M / DIM_TC];
		if constexpr (C_EXIST) {
			if (bm < n) {
				const auto real_m = min(DIM_BLOCK_M, n - bm);
				copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N>(
						smem_A_ptr,
						gmem_C_ptr + bm, ld_C,
						real_m, n
						);
				__syncthreads();
				for (auto sn = decltype(DIM_BLOCK_M)(0); sn < DIM_BLOCK_M; sn += DIM_TC) {
					auto load_ptr = smem_A_ptr + DIM_BLOCK_M * DIM_TC * cutf::thread::get_warp_id() + sn;
					nvcuda::wmma::load_matrix_sync(frag_C[sn], load_ptr, DIM_BLOCK_M, nvcuda::wmma::mem_col_major);
					mtk::wmma::fill_zero(frag_d_C[sn]);
				}
			} else {
				for (auto sn = decltype(DIM_BLOCK_M)(0); sn < DIM_BLOCK_M; sn += DIM_TC) {
					mtk::wmma::fill_zero(frag_C[sn]);
					mtk::wmma::fill_zero(frag_d_C[sn]);
				}
			}
		} else {
			for (auto sn = decltype(DIM_BLOCK_M)(0); sn < DIM_BLOCK_M; sn += DIM_TC) {
				mtk::wmma::fill_zero(frag_C[sn]);
				mtk::wmma::fill_zero(frag_d_C[sn]);
			}
		}
		const auto real_m = min(DIM_BLOCK_M, n - bm);
		copy_matrix_g2s_XPxN<block_size, DIM_BLOCK_M, DIM_N>(
				smem_A_ptr,
				gmem_C_ptr + bm, ld_C,
				real_m, n
				);
		__syncthreads();
		for (auto bk = decltype(DIM_N)(0); bk < DIM_N; bk += K_BLOCKING) {
			nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, DIM_TC, DIM_TC, DIM_TC, half, nvcuda::wmma::col_major> frag_B[NUM_BLOCKINGS], frag_d_B[NUM_BLOCKINGS];
			const auto b_offset = cutf::thread::get_warp_id() * DIM_N * DIM_TC + bk;
			mtk::wmma::foreach<decltype(frag_B[0])>(
				[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
					const auto mem_m = mem_index % DIM_TC;
					const auto mem_n = mem_index / DIM_TC;
					const auto m_offset = b_offset + mem_n * DIM_N + mem_m;
					for (unsigned k = 0; k < NUM_BLOCKINGS; k++) {
						auto v = smem_B_ptr[m_offset + k * DIM_TC];
						if constexpr (A_MINUS) {
							v *= -1.f;
						}
						const auto hv = cutf::type::cast<half>(v);
						const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
						for (unsigned i = 0; i < fragment_index_count; i++) {
							const auto frag_index = frag_index_list[i];
							frag_B[k].x[frag_index] = hv;
							frag_d_B[k].x[frag_index] = dhv;
						}
					}
				});
			for (auto sn = decltype(DIM_BLOCK_M)(0); sn < DIM_BLOCK_M; sn += DIM_TC) {
				nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, DIM_TC, DIM_TC, DIM_TC, half, nvcuda::wmma::col_major> frag_A[NUM_BLOCKINGS], frag_d_A[NUM_BLOCKINGS];
				const auto a_offset = (bm + sn) * DIM_N + bk;
				mtk::wmma::foreach<decltype(frag_A[0])>(
					[&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned mem_index) {
						const auto mem_n = mem_index % DIM_TC;
						const auto mem_m = mem_index / DIM_TC;
						const auto m_offset = a_offset + mem_m * DIM_N + mem_n;
						for (unsigned k = 0; k < NUM_BLOCKINGS; k++) {
							const auto v = smem_A_ptr[m_offset + k * DIM_TC];
							const auto hv = cutf::type::cast<half>(v);
							const auto dhv = cutf::type::cast<half>((v - cutf::type::cast<float>(hv)) * cor_scale);
							for (unsigned i = 0; i < fragment_index_count; i++) {
								const auto frag_index = frag_index_list[i];
								frag_A[k].x[frag_index] = hv;
								frag_d_A[k].x[frag_index] = dhv;
							}
						}
					});

				const auto c_index = sn / DIM_TC;
				for (unsigned k = 0; k < NUM_BLOCKINGS; k++) {
					nvcuda::wmma::mma_sync(frag_d_C[c_index], frag_A  [k], frag_d_B[k], frag_d_C[c_index]);
					nvcuda::wmma::mma_sync(frag_d_C[c_index], frag_d_A[k], frag_B  [k], frag_d_C[c_index]);
					nvcuda::wmma::mma_sync(frag_C  [c_index], frag_A  [k], frag_B  [k], frag_C  [c_index]);
				}
			}
		}
		__syncthreads();
		for (auto sn = decltype(DIM_BLOCK_M)(0); sn < DIM_BLOCK_M; sn += DIM_TC) {
			const auto c_index = sn / DIM_TC;
			for (unsigned i = 0; i < frag_C[0].num_elements; i++) {
				frag_C[c_index].x[i] += frag_d_C[c_index].x[i] / cor_scale;
			}
			auto store_ptr = smem_A_ptr + DIM_BLOCK_M * DIM_TC * cutf::thread::get_warp_id() + sn;
			nvcuda::wmma::store_matrix_sync(store_ptr, frag_C[c_index], DIM_BLOCK_M, nvcuda::wmma::mem_col_major);
		}
		__syncthreads();
		copy_matrix_s2g_XPxN<block_size, DIM_BLOCK_M, DIM_N>(
				gmem_D_ptr + bm, ld_D,
				smem_A_ptr,
				real_m, n
				);
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode, int A_MINUS, int B_TRANS, int C_EXIST>
__device__ void gemm_MxNxN_core(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_D_ptr, const std::size_t ld_D,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_A_ptr, const std::size_t ld_A,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_B_ptr, const std::size_t ld_B,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_C_ptr, const std::size_t ld_C,
		const std::size_t m, const std::size_t n
		) {
	if constexpr (compute_mode == mtk::tsqr_tc::compute_mode::fp32_hmma_cor) {
		gemm_MxNxN_core_fp32_hmma_cor<A_MINUS, B_TRANS, C_EXIST>(
				gmem_D_ptr, ld_D,
				gmem_A_ptr, ld_A,
				gmem_B_ptr, ld_B,
				gmem_C_ptr, ld_C,
				m, n
				);
	}
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
__global__ void tsqr_backward_kernel(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_OUT_Q_ptr  , const std::size_t ld_OUT_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_W_ptr      , const std::size_t ld_W,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_Y_ptr      , const std::size_t ld_Y,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_IN_Q_ptr   , const std::size_t ld_IN_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_WORKING_ptr,
		const std::size_t n,
		const std::size_t* const m_start_list
		) {
	const auto matrix_id = blockIdx.x;
	const auto start_m = (m_start_list == nullptr) ? matrix_id       * 2 * n : m_start_list[matrix_id];
	const auto end_m   = (m_start_list == nullptr) ? (matrix_id + 1) * 2 * n : m_start_list[matrix_id + 1];
	const auto m = end_m - start_m;

	auto g_Y_ptr = gmem_Y_ptr + start_m;
	auto g_W_ptr = gmem_W_ptr + start_m;
	auto g_OUT_Q_ptr = gmem_OUT_Q_ptr + start_m;
	auto g_IN_Q_ptr = gmem_IN_Q_ptr + matrix_id * n;
	auto working_memory_ptr = gmem_WORKING_ptr + matrix_id * n * n;

	gemm_MxNxN_core<compute_mode, 0, 1, 0>(
			working_memory_ptr, n,
			g_Y_ptr, ld_Y,
			g_IN_Q_ptr, ld_IN_Q,
			nullptr, 0,
			n, n
			);
	__syncthreads();
	gemm_MxNxN_core<compute_mode, 1, 0, 1>(
			g_OUT_Q_ptr, ld_OUT_Q,
			g_W_ptr, ld_W,
			working_memory_ptr, n,
			g_IN_Q_ptr, ld_IN_Q,
			m, n
			);
}

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void tsqr_backward(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_OUT_Q_ptr  , const std::size_t ld_OUT_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_W_ptr      , const std::size_t ld_W,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_Y_ptr      , const std::size_t ld_Y,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_IN_Q_ptr   , const std::size_t ld_IN_Q,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const gmem_WORKING_ptr,
		const std::size_t n,
		const std::size_t split_size,
		const std::size_t* const m_start_list,
		const cudaStream_t cuda_stream
		) {
	const auto block_size = 256;

	tsqr_backward_kernel<compute_mode><<<split_size, block_size, 0, cuda_stream>>>(
			gmem_OUT_Q_ptr  , ld_OUT_Q,
			gmem_W_ptr      , ld_W,
			gmem_Y_ptr      , ld_Y,
			gmem_IN_Q_ptr   , ld_IN_Q,
			gmem_WORKING_ptr,
			n, m_start_list
			);
}

template <class T>
__global__ void make_identity_matrix_kernel(
		T* const ptr,
		unsigned m, unsigned n
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (m * n <= tid) {
		return;
	}

	T v = cutf::type::cast<T>(0.0f);
	if (tid / m == tid % m) {
		v = cutf::type::cast<T>(1.0f);
	}
	ptr[tid] = v;
}

template <class T>
void make_indentity_matrix(
		T* const ptr,
		const unsigned m, const unsigned n,
		const cudaStream_t cuda_stream
		) {
	const auto block_size = 256;

	make_identity_matrix_kernel<<<(m * n + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
			ptr,
			m, n
			);
}

} // noname namespace

template <mtk::tsqr_tc::compute_mode::type compute_mode>
void mtk::tsqr_tc::tsqr(
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const r_ptr, const std::size_t ld_R,
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const q_ptr, const std::size_t ld_Q,
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const a_ptr, const std::size_t ld_A,
		const std::size_t m, const std::size_t n,
		mtk::tsqr_tc::tsqr_buffer<compute_mode>& buffer,
		const cudaStream_t cuda_stream
		) {
	assert(n <= 128);

	for (std::size_t i = 0; i < buffer.get_split_count() + 1; i++) {
		buffer.get_index_buffer_ptr()[i] = i * m / buffer.get_split_count();
	}
	cutf::memory::copy_async(buffer.get_index_buffer_ptr(), buffer.get_index_buffer_host_ptr(), buffer.get_index_buffer_count(), cuda_stream);
	for (std::size_t i = 0; i < buffer.get_split_count(); i++) {
		buffer.get_index_buffer_ptr()[i] = i * (2 * n);
	}

	unsigned r_ptr_index = 0;
	typename mtk::tsqr_tc::tsqr_buffer<compute_mode>::buffer_type* r_buffer_list[2];
	r_buffer_list[0] = buffer.get_r_buffer_ptr();
	r_buffer_list[1] = buffer.get_r_buffer_ptr() + buffer.get_split_count() * n * n;

	std::size_t wy_ptr_offset = 0;

	// Forwad computation
	mtk::tsqr_tc::qr256x128_batched<compute_mode>(
			buffer.get_w_buffer_ptr() + wy_ptr_offset, m,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, m,
			r_buffer_list[r_ptr_index], n * buffer.get_split_count(),
			a_ptr, ld_A,
			n,
			buffer.get_split_count(),
			buffer.get_index_buffer_ptr(),
			cuda_stream
			);
	wy_ptr_offset += m * n;
	r_ptr_index = 1 - r_ptr_index;

	cutf::memory::copy_async(buffer.get_index_buffer_ptr(), buffer.get_index_buffer_host_ptr(), buffer.get_index_buffer_count(), cuda_stream);

	for (std::size_t s = buffer.get_split_count(); s > 1; s >>= 1) {
		mtk::tsqr_tc::qr256x128_batched<compute_mode>(
				buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				r_buffer_list[r_ptr_index], n * s,
				r_buffer_list[1 - r_ptr_index], 2 * n * s,
				n,
				s,
				buffer.get_index_buffer_ptr(),
				cuda_stream
				);
		wy_ptr_offset += 2 * n * n * s;
		r_ptr_index = 1 - r_ptr_index;
	}
	mtk::tsqr_tc::qr256x128<compute_mode>(
			buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n,
			r_ptr, ld_R,
			r_buffer_list[1 - r_ptr_index], 2 * n,
			2 * n, n,
			cuda_stream
			);
	// Backward computation
	make_indentity_matrix(
			buffer.get_r_buffer_ptr(),
			2 * n, n,
			cuda_stream
			);
	tsqr_backward<compute_mode>(
			buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n,
			buffer.get_r_buffer_ptr(), n,
			buffer.get_r_buffer_ptr(),
			n, 1,
			buffer.get_index_buffer_ptr(),
			cuda_stream
			);
	for (std::size_t s = 2; s < buffer.get_split_count(); s <<= 1) {
		const auto prev_wy_ptr_offset = wy_ptr_offset;
		wy_ptr_offset -= 2 * n * n * s;
		tsqr_backward<compute_mode>(
				buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_w_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_y_buffer_ptr() + wy_ptr_offset, 2 * n * s,
				buffer.get_w_buffer_ptr() + prev_wy_ptr_offset, 2 * n * s / 2,
				buffer.get_r_buffer_ptr(),
				n, s,
				buffer.get_index_buffer_ptr(),
				cuda_stream
				);
	}
	for (std::size_t i = 0; i < buffer.get_split_count() + 1; i++) {
		buffer.get_index_buffer_ptr()[i] = i * m / buffer.get_split_count();
	}
	cutf::memory::copy_async(buffer.get_index_buffer_ptr(), buffer.get_index_buffer_host_ptr(), buffer.get_index_buffer_count(), cuda_stream);
	const auto prev_wy_ptr_offset = wy_ptr_offset;
	wy_ptr_offset -= m * n;
	const auto s = buffer.get_split_count();
	tsqr_backward<compute_mode>(
			q_ptr, ld_Q,
			buffer.get_w_buffer_ptr() + wy_ptr_offset, m,
			buffer.get_y_buffer_ptr() + wy_ptr_offset, m,
			buffer.get_w_buffer_ptr() + prev_wy_ptr_offset, 2 * n * s / 2,
			buffer.get_r_buffer_ptr(),
			n, s,
			buffer.get_index_buffer_ptr(),
			cuda_stream
			);
}

#define MTK_INSTANCE_TSQR(compute_mode) \
template void mtk::tsqr_tc::tsqr(\
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const typename mtk::tsqr_tc::detail::get_type<compute_mode>::type* const, const std::size_t, \
		const std::size_t, const std::size_t, \
		mtk::tsqr_tc::tsqr_buffer<compute_mode>&, \
		const cudaStream_t\
		)

MTK_INSTANCE_TSQR(mtk::tsqr_tc::compute_mode::fp32_hmma_cor);
