#include <cstdint>
#include <cutf/type.hpp>

namespace {

constexpr unsigned warp_size = 32u;

template <unsigned block_size, unsigned size, class T>
__device__ void fill_zero(T* const ptr) {
	if constexpr (size % block_size == 0) {
		for (unsigned i = 0; i < size; i += block_size) {
			const auto index = i + threadIdx.x;
			ptr[index] = cutf::type::cast<T>(0.0f);
		}
	} else {
		for (unsigned index = threadIdx.x; index < size; index += block_size) {
			ptr[index] = cutf::type::cast<T>(0.0f);
		}
	}
}
} // noname namespace
